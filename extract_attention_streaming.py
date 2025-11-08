#!/usr/bin/env python3
"""
StreamingVLM Attention Extraction

Extracts and visualizes multimodal attention maps from StreamingVLM's first decoder layer.

Features:
- Text→Vision attention (averaged across heads)
- Vision→Text attention (averaged across heads)
- Vision→Text per-head extraction (optional, no averaging)
- Special token filtering (remove <|video_pad|> etc.)
- Per-head visualization with row normalization
- Top-k token analysis per vision patch

Usage:
    # Basic usage (Text→Vision and Vision→Text averaged)
    python extract_attention_streaming.py \
        --model_path mit-han-lab/StreamingVLM \
        --model_base Qwen2_5 \
        --pos_mode shrink \
        --video_path video.mp4 \
        --question "What is happening?" \
        --output_dir outputs/test

    # Advanced: Per-head extraction with filtering and visualization
    python extract_attention_streaming.py \
        --model_path mit-han-lab/StreamingVLM \
        --model_base Qwen2_5 \
        --video_path video.mp4 \
        --question "What is happening?" \
        --output_dir outputs/test \
        --vision_to_text \
        --filter_video_pad \
        --visualize_per_head \
        --max_vis_heads 8 \
        --save_top_k_analysis \
        --top_k 10 \
        --max_q_tokens 50 \
        --max_frames 5

Output Structure:
    outputs/test/
    ├── prefill_attention.pt                          # Full attention [B, H, Q, K] all heads
    ├── frame_000_text_to_vision.npy                 # Text→Vision [T, V] averaged over heads
    ├── frame_000_vision_to_text.npy                 # Vision→Text [V, T] averaged over heads
    ├── frame_000_vision_to_text_head00.npy          # Vision→Text [V, T] head 0 (if --vision_to_text)
    ├── frame_000_vision_to_text_head01.npy          # Vision→Text [V, T] head 1 (if --vision_to_text)
    ├── frame_000_head00_vision_to_text.png          # Per-head visualization (if --visualize_per_head)
    ├── top_k_analysis/
    │   └── frame_000_head00_topk.json               # Top-k tokens per patch (if --save_top_k_analysis)
    └── metadata.json                                 # Extraction metadata
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import math
from typing import Dict, List, Tuple, Optional
from typing import Union

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.qwen2.patch_model import convert_qwen2_to_streaming
from streaming_vlm.inference.streaming_args import StreamingArgs

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("Warning: qwen_vl_utils not found")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_model_and_processor(model_path, model_base='Qwen2_5'):
    """Load model and processor based on model_base (Qwen2 or Qwen2_5)."""
    if model_base == 'Qwen2_5':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        model = convert_qwen2_5_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    elif model_base == 'Qwen2':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        model = convert_qwen2_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor


class StreamingAttentionCapture:
    """Captures attention from streaming inference during prefill and decode phases."""

    def __init__(self, capture_decode: bool = False, max_decode_steps: int = 4):
        self.prefill_attention = None
        self.decode_attentions = []
        self.capture_decode = capture_decode
        self.max_decode_steps = max_decode_steps
        self.hook_handles = []
        self.is_prefill = True
        self.decode_count = 0

    def register_hook(self, model, layer_idx: int = 0):
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    bsz, num_heads, q_len, k_len = attn_weights.shape
                    if self.is_prefill and q_len > 1:
                        self.prefill_attention = attn_weights.detach()
                        self.is_prefill = False
                    elif self.capture_decode and self.decode_count < self.max_decode_steps:
                        self.decode_attentions.append(attn_weights.detach())
                        self.decode_count += 1

        target_module = model.model.language_model.layers[layer_idx].self_attn
        handle = target_module.register_forward_hook(attention_hook)
        self.hook_handles.append(handle)

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


class TokenIndexMapper:
    """Maps token indices to vision/text categories and per-frame boundaries."""

    def __init__(self, input_ids, video_token_id, video_grid_thw):
        self.input_ids = input_ids
        self.video_token_id = video_token_id
        self.video_grid_thw = video_grid_thw
        self.device = input_ids.device

        video_mask = (input_ids[0] == video_token_id)
        self.vision_idx = torch.where(video_mask)[0].to(self.device)

        if len(video_grid_thw.shape) == 2:
            t, h, w = video_grid_thw[0]
        else:
            t, h, w = video_grid_thw

        self.num_frames = int(t.item())
        self.pre_merger_grid = (int(h.item()), int(w.item()))
        pre_merger_patches = int(h.item()) * int(w.item())

        total_vision_tokens = len(self.vision_idx)
        assert total_vision_tokens > 0, "No vision tokens found"

        self.patches_per_frame = total_vision_tokens // self.num_frames
        self.spatial_grid = self._infer_spatial_grid(self.patches_per_frame)

    def _infer_spatial_grid(self, patches_per_frame: int) -> Tuple[int, int]:
        side = int(round(math.sqrt(patches_per_frame)))
        if side * side == patches_per_frame:
            return (side, side)
        pre_h, pre_w = self.pre_merger_grid
        for merger_side in range(1, 10):
            post_h = pre_h // merger_side
            post_w = pre_w // merger_side
            if post_h * post_w == patches_per_frame:
                return (post_h, post_w)
        return (side, patches_per_frame // side)

    def get_frame_indices(self, frame_idx: int) -> torch.Tensor:
        start = frame_idx * self.patches_per_frame
        end = start + self.patches_per_frame
        return self.vision_idx[start:end]

    def get_question_tokens(self, processor, filter_special: bool = False, filter_video_pad: bool = False) -> torch.Tensor:
        """Get question token indices, optionally filtering special tokens.

        Args:
            processor: Tokenizer processor
            filter_special: If True, filter out special tokens (im_start, im_end, etc.)
            filter_video_pad: If True, also filter out video_pad tokens
        """
        ids = self.input_ids[0]
        dev = self.device

        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = processor.tokenizer.convert_tokens_to_ids("user")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
        nl_id = processor.tokenizer.convert_tokens_to_ids("\n")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

        text_start, text_end = None, None
        for i in range(len(ids)):
            if ids[i].item() == im_start_id:
                s = i
                e = i + 1
                while e < len(ids) and ids[e].item() != im_end_id:
                    e += 1
                if s + 1 < len(ids) and ids[s + 1].item() == user_id and e > s:
                    t0 = s + 2
                    if t0 < len(ids) and ids[t0].item() == nl_id:
                        t0 += 1
                    text_start, text_end = t0, e
                    break

        if text_start is None:
            text_start = (self.vision_idx[-1].item() + 1) if len(self.vision_idx) > 0 else 0
            text_end = self.input_ids.shape[1]

        # Build exclusion set
        exclusions = set()
        if filter_special:
            exclusions.update([im_start_id, im_end_id, user_id, assistant_id, nl_id])
        else:
            # Always filter these basic special tokens
            exclusions.update([im_start_id, im_end_id, user_id, assistant_id, nl_id])

        if filter_video_pad:
            exclusions.add(video_pad_id)

        keep = []
        for i in range(text_start, text_end):
            tok = ids[i].item()
            if tok not in exclusions:
                keep.append(i)

        return torch.tensor(keep, device=dev, dtype=torch.long)


def analyze_top_k_tokens(attention_matrix: np.ndarray, text_tokens: List[str], k: int = 5) -> List[Dict]:
    """Analyze top-k attended tokens for each vision patch.

    Args:
        attention_matrix: [V, T] numpy array
        text_tokens: List of decoded text tokens
        k: Number of top tokens to extract per patch

    Returns:
        List of dicts with {patch_idx, top_k_indices, top_k_tokens, top_k_weights}
    """
    results = []
    num_patches = attention_matrix.shape[0]

    for patch_idx in range(num_patches):
        patch_attn = attention_matrix[patch_idx]
        top_k_idx = np.argsort(patch_attn)[-k:][::-1]  # Descending order
        top_k_weights = patch_attn[top_k_idx]
        top_k_tokens_list = [text_tokens[i] if i < len(text_tokens) else f"<idx_{i}>" for i in top_k_idx]

        results.append({
            'patch_idx': patch_idx,
            'top_k_indices': top_k_idx.tolist(),
            'top_k_tokens': top_k_tokens_list,
            'top_k_weights': top_k_weights.tolist()
        })

    return results


class AttentionVisualizer:
    """Visualizes attention patterns as heatmaps and videos."""

    def __init__(self, output_dir: str = "outputs/attn_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_attention_matrix(
        self,
        attention: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        y_tokens: List[str] = None,
        x_tokens: List[str] = None,
        spatial_grid: Tuple[int, int] = None,
        spatial_axis: int = None,
        save_path: str = None
    ):
        if spatial_grid is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None

        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(title)

        if y_tokens and len(y_tokens) <= 20:
            ax1.set_yticks(range(len(y_tokens)))
            ax1.set_yticklabels(y_tokens, rotation=0)
        if x_tokens and len(x_tokens) <= 20:
            ax1.set_xticks(range(len(x_tokens)))
            ax1.set_xticklabels(x_tokens, rotation=45, ha='right')

        if spatial_grid is not None and spatial_axis is not None:
            h, w = spatial_grid
            spatial_attn = attention.mean(axis=spatial_axis).reshape(h, w)

            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=ax2, cbar=True, square=True)
            else:
                im2 = ax2.imshow(spatial_attn, cmap='hot')
                plt.colorbar(im2, ax=ax2)

            ax2.set_xlabel('Patch X')
            ax2.set_ylabel('Patch Y')
            avg_label = 'vision patches' if spatial_axis == 0 else 'text tokens'
            ax2.set_title(f'Spatial Attention (Avg over {avg_label})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "attention_matrix.png", dpi=150, bbox_inches='tight')

        plt.close()

    def plot_frame_to_text_attention(
        self,
        attention: np.ndarray,
        frame_idx: int,
        spatial_grid: Tuple[int, int],
        text_tokens: List[str],
        save_path: str = None
    ):
        h, w = spatial_grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel('Vision Patch Index')
        ax1.set_ylabel('Text Token Index')
        ax1.set_title(f'Frame {frame_idx}: Text→Vision Attention')

        if len(text_tokens) <= 20:
            ax1.set_yticks(range(len(text_tokens)))
            ax1.set_yticklabels(text_tokens, rotation=0)

        spatial_attn = attention.mean(axis=0).reshape(h, w)

        if HAS_SEABORN:
            sns.heatmap(spatial_attn, cmap='hot', ax=ax2, cbar=True, square=True)
        else:
            im2 = ax2.imshow(spatial_attn, cmap='hot')
            plt.colorbar(im2, ax=ax2)

        ax2.set_xlabel('Patch X')
        ax2.set_ylabel('Patch Y')
        ax2.set_title(f'Frame {frame_idx}: Spatial Attention (Avg over Text)')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"frame_{frame_idx:03d}_attention.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_matrix_no_avg(
        self,
        mat: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        save_path: str
    ):
        """Plot raw attention matrix without averaging."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if HAS_SEABORN:
            sns.heatmap(mat, cmap='hot', ax=ax, cbar=True, vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(mat, aspect='auto', origin='lower', cmap='hot', vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_all_frames_summary(
        self,
        frame_attentions: List[np.ndarray],
        spatial_grid: Tuple[int, int],
        save_path: str = None
    ):
        h, w = spatial_grid
        num_frames = len(frame_attentions)

        ncols = min(num_frames, 8)
        nrows = (num_frames + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        if num_frames == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_frames > 1 else axes

        for idx, attn in enumerate(frame_attentions):
            spatial_attn = attn.mean(axis=0).reshape(h, w)
            ax = axes[idx] if num_frames > 1 else axes[0]

            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=ax, cbar=False, square=True)
            else:
                ax.imshow(spatial_attn, cmap='hot')

            ax.set_title(f'Frame {idx}', fontsize=10)
            ax.axis('off')

        for idx in range(num_frames, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "all_frames_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_vision_to_text_per_head(
        self,
        attention_head: np.ndarray,
        frame_idx: int,
        head_idx: int,
        text_tokens: List[str],
        spatial_grid: Tuple[int, int],
        normalize_rows: bool = True,
        save_path: str = None
    ):
        """Plot Vision→Text attention for a single head with improved clarity.

        Args:
            attention_head: [V, T] attention matrix for one head
            frame_idx: Frame index
            head_idx: Head index
            text_tokens: List of decoded text tokens
            spatial_grid: (H, W) spatial grid for vision patches
            normalize_rows: If True, normalize each row to sum to 1
            save_path: Output path
        """
        h, w = spatial_grid

        # Normalize if requested
        attn = attention_head.copy()
        if normalize_rows:
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            attn = attn / row_sums

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Full heatmap
        if HAS_SEABORN:
            sns.heatmap(attn, cmap='hot', ax=ax1, cbar=True, vmin=0, vmax=attn.max())
        else:
            im1 = ax1.imshow(attn, cmap='hot', aspect='auto', vmin=0, vmax=attn.max())
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel('Text Token Index')
        ax1.set_ylabel('Vision Patch Index')
        norm_str = " (row-normalized)" if normalize_rows else ""
        ax1.set_title(f'Frame {frame_idx} Head {head_idx}: Vision→Text{norm_str}')

        # Show token labels if not too many
        if len(text_tokens) <= 50:
            ax1.set_xticks(range(len(text_tokens)))
            ax1.set_xticklabels(text_tokens, rotation=90, ha='right', fontsize=6)

        # Right: Spatial view (averaged over text tokens)
        spatial_attn = attn.mean(axis=1).reshape(h, w)
        if HAS_SEABORN:
            sns.heatmap(spatial_attn, cmap='hot', ax=ax2, cbar=True, square=True)
        else:
            im2 = ax2.imshow(spatial_attn, cmap='hot')
            plt.colorbar(im2, ax=ax2)

        ax2.set_xlabel('Patch X')
        ax2.set_ylabel('Patch Y')
        ax2.set_title(f'Spatial View (avg over text)')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"frame_{frame_idx:03d}_head{head_idx:02d}_vision_to_text.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path


def extract_attention_streaming(
    model_path: str,
    video_path: str,
    question: str,
    output_dir: str,
    model_base: str = 'Qwen2_5',
    pos_mode: str = "shrink",
    all_text: bool = False,
    device: str = "cuda:0",
    capture_decode: bool = False,
    max_decode_steps: int = 4,
    vision_to_text: bool = False,
    max_q_tokens: Optional[int] = None,
    max_frames: Optional[int] = None,
    filter_video_pad: bool = False,
    visualize_per_head: bool = False,
    max_vis_heads: Optional[int] = None,
    save_top_k_analysis: bool = False,
    top_k: int = 5,
    normalize_rows: bool = True,
    fps: float = 1.0,
    streaming_args: Optional[StreamingArgs] = None
) -> Dict:
    """Extract attention maps from StreamingVLM inference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STREAMING ATTENTION EXTRACTION")
    print("=" * 80)

    # Load model and processor
    print(f"\n[1/6] Loading model: {model_path} (base: {model_base})")
    model, processor = load_model_and_processor(model_path, model_base)
    print(f"Device: {device}")

    # Initialize streaming args
    print(f"[2/6] Initializing streaming args (pos_mode={pos_mode}, all_text={all_text})")
    if streaming_args is None:
        streaming_args = StreamingArgs(pos_mode=pos_mode, all_text=all_text)

    # Register attention hook
    print(f"[3/6] Registering attention hook (capture_decode={capture_decode})")
    capture = StreamingAttentionCapture(capture_decode, max_decode_steps)
    capture.register_hook(model, layer_idx=0)

    # Process video
    print(f"[4/6] Processing video: {video_path}")
    print(f"Question: {question}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps, "max_pixels": 40000},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if HAS_QWEN_VL_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        raise ImportError("qwen_vl_utils is required")

    # Save the sampled frames exactly as constructed for the processor
    def _frame_to_hwc_uint8(frame: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert a single frame to HWC uint8 safely, handling 0-1 vs 0-255 floats.

        - Accepts CHW or HWC
        - If float and max>2.0, assume 0-255; else assume 0-1
        - If integer but not uint8, clip to [0,255] and cast
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        # Accept shapes: HWC or CHW
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[2] not in (1, 3):
            # Likely CHW -> convert to HWC
            frame = np.transpose(frame, (1, 2, 0))
        # Handle dtype/range
        if np.issubdtype(frame.dtype, np.floating):
            fmin = float(np.nanmin(frame)) if frame.size else 0.0
            fmax = float(np.nanmax(frame)) if frame.size else 0.0
            if fmax > 2.0:
                arr = np.clip(frame, 0.0, 255.0).astype(np.uint8)
            else:
                arr = np.clip(frame, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
        elif frame.dtype == np.uint8:
            arr = frame
        else:
            # Other integer types
            arr = np.clip(frame, 0, 255).astype(np.uint8)
        return arr

    sampled_dir = output_dir / "sampled_frames"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    sampled_count = 0
    try:
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) > 0:
            vid = video_inputs[0]
            if isinstance(vid, (list, tuple)):
                from PIL import Image
                for i, f in enumerate(vid):
                    out_path = sampled_dir / f"frame_{i:05d}.jpg"
                    if hasattr(f, 'save'):
                        # PIL.Image
                        if i == 0:
                            try:
                                npf = np.array(f)
                                print(f"[sampled_frames] frame_00000 stats: dtype={npf.dtype}, min={npf.min() if npf.size else 'n/a'}, max={npf.max() if npf.size else 'n/a'}")
                            except Exception:
                                pass
                        f.save(out_path)
                    else:
                        # Tensor/ndarray
                        if isinstance(f, torch.Tensor):
                            npf = f.detach().cpu().numpy()
                        else:
                            npf = np.array(f)
                        if i == 0:
                            try:
                                print(f"[sampled_frames] frame_00000 stats: dtype={npf.dtype}, min={np.min(npf) if npf.size else 'n/a'}, max={np.max(npf) if npf.size else 'n/a'}")
                            except Exception:
                                pass
                        arr = _frame_to_hwc_uint8(f)
                        Image.fromarray(arr).save(out_path)
                    sampled_count += 1
            elif isinstance(vid, (torch.Tensor, np.ndarray)):
                from PIL import Image
                tdim = vid.shape[0]
                for i in range(tdim):
                    out_path = sampled_dir / f"frame_{i:05d}.jpg"
                    fr = vid[i]
                    if isinstance(fr, torch.Tensor):
                        npf = fr.detach().cpu().numpy()
                    else:
                        npf = np.array(fr)
                    if i == 0:
                        try:
                            print(f"[sampled_frames] frame_00000 stats: dtype={npf.dtype}, min={np.min(npf) if npf.size else 'n/a'}, max={np.max(npf) if npf.size else 'n/a'}")
                        except Exception:
                            pass
                    arr = _frame_to_hwc_uint8(fr)
                    Image.fromarray(arr).save(out_path)
                sampled_count = tdim
        # Save metadata about sampling
        (output_dir / "sampled_frames_metadata.json").write_text(
            json.dumps({
                "video_path": str(video_path),
                "fps": fps,
                "num_sampled_frames": sampled_count
            }, indent=2)
        )
    except Exception as e:
        print(f"Warning: failed to export sampled frames: {type(e).__name__}: {e}")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    print(f"Input shape: {inputs.input_ids.shape}")
    print(f"Video grid THW: {inputs.video_grid_thw}")

    # Populate streaming_args with input information (required for shrink mode)
    if streaming_args.pos_mode == "shrink":
        streaming_args.input_ids = inputs['input_ids']
        streaming_args.video_grid_thw = inputs['video_grid_thw']
        streaming_args.second_per_grid_ts = inputs.get('second_per_grid_ts', None)

    # Generate
    print(f"[5/6] Generating with StreamingVLM")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            use_cache=True,
            return_dict_in_generate=True,
            streaming_args=streaming_args,
            pad_token_id=151645,
            output_attentions=True
        )

    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nGenerated: {generated_text[0]}")

    capture.clear_hooks()

    # Extract attention
    print(f"[6/6] Extracting and visualizing attention\n")

    if capture.prefill_attention is None:
        raise RuntimeError("No prefill attention captured")

    attn_probs = capture.prefill_attention

    print(f"Prefill attention shape: {attn_probs.shape}")

    # Get token mapping
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    mapper = TokenIndexMapper(inputs.input_ids, video_token_id, inputs.video_grid_thw)

    pre_h, pre_w = mapper.pre_merger_grid
    pre_patches = pre_h * pre_w

    print(f"Video structure: {mapper.num_frames} frames, ({pre_h}, {pre_w}) grid = {pre_patches} patches/frame")
    if mapper.patches_per_frame != pre_patches:
        print(f"Merger applied: {pre_patches} -> {mapper.patches_per_frame} tokens (spatial grid {mapper.spatial_grid})")

    # Average attention over heads
    attn_mean = attn_probs.mean(dim=1).squeeze(0)
    print(f"Attention mean shape: {attn_mean.shape}")

    # Get question tokens (with optional video_pad filtering)
    text_idx = mapper.get_question_tokens(processor, filter_video_pad=filter_video_pad)
    question_tokens = [processor.tokenizer.decode([t]) for t in inputs.input_ids[0][text_idx]]

    filter_msg = " (video_pad filtered)" if filter_video_pad else ""
    print(f"Question tokens: {len(text_idx)}{filter_msg}")
    print(f"Text range: [{text_idx[0].item()}, {text_idx[-1].item()}]")

    # Verbose: print each text token with its absolute index and decoded piece
    try:
        print("\nText tokens (absolute_index -> piece):")
        ids_row = inputs.input_ids[0]
        for j in text_idx.tolist():
            tok_id = int(ids_row[j])
            piece = processor.tokenizer.decode([tok_id])
            print(f"{j}: {piece!r}")

        # Also print text-only row mapping and save CSV for easy lookup in plots
        print("\nText-only index mapping (row -> abs_index, piece):")
        mapping_rows = []
        for r, abs_idx in enumerate(text_idx.tolist()):
            tok_id = int(ids_row[abs_idx])
            piece = processor.tokenizer.decode([tok_id])
            print(f"{r} -> {abs_idx}, {piece!r}")
            mapping_rows.append({
                "row": r,
                "abs_index": abs_idx,
                "token_id": tok_id,
                "piece": piece
            })
        # Save CSV mapping
        try:
            import csv
            with open(output_dir / "text_token_map.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["row", "abs_index", "token_id", "piece"])
                writer.writeheader()
                for row in mapping_rows:
                    writer.writerow(row)
            print(f"Saved text token map to {output_dir / 'text_token_map.csv'}")
        except Exception as e:
            print(f"Warning: failed to save text_token_map.csv: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Warning: failed to print text tokens: {type(e).__name__}: {e}")

    # Extract per-frame attention
    frame_attentions_tv = []
    frame_attentions_vt = []

    print(f"\nExtracting per-frame attention (upsampling to pre-merger grid)")

    for frame_idx in range(mapper.num_frames):
        frame_vision_idx = mapper.get_frame_indices(frame_idx)

        # TEXT→VISION
        frame_attn_tv = torch.index_select(attn_mean, 0, text_idx)
        frame_attn_tv = torch.index_select(frame_attn_tv, 1, frame_vision_idx)
        frame_attn_tv = torch.nn.functional.normalize(frame_attn_tv, p=1, dim=1)

        # VISION→TEXT
        frame_attn_vt = torch.index_select(attn_mean, 0, frame_vision_idx)
        frame_attn_vt = torch.index_select(frame_attn_vt, 1, text_idx)
        frame_attn_vt = torch.nn.functional.normalize(frame_attn_vt, p=1, dim=1)

        # Upsample if merger was applied
        nv_merged = mapper.patches_per_frame
        if nv_merged != pre_patches:
            ratio = pre_patches // nv_merged
            merger_side = int(round(math.sqrt(ratio)))
            assert merger_side * merger_side * nv_merged == pre_patches

            # Upsample TEXT→VISION
            post_h, post_w = mapper.spatial_grid
            attn_hw = frame_attn_tv.view(frame_attn_tv.size(0), post_h, post_w)
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=1)
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=2)
            frame_attn_tv = attn_hw.view(frame_attn_tv.size(0), pre_patches)

            # Upsample VISION→TEXT
            attn_hw_vt = frame_attn_vt.view(post_h, post_w, frame_attn_vt.size(1))
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=0)
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=1)
            frame_attn_vt = attn_hw_vt.view(pre_patches, frame_attn_vt.size(1))

        # Convert to numpy
        attn_tv_np = frame_attn_tv.detach().float().cpu().numpy()
        attn_vt_np = frame_attn_vt.detach().float().cpu().numpy()

        frame_attentions_tv.append(attn_tv_np)
        frame_attentions_vt.append(attn_vt_np)

    # Text→Vision Per-Head Extraction (No Averaging)
    print(f"\nExtracting Text→Vision per-head (no averaging)")
    num_heads_tv = attn_probs.shape[1]
    actual_num_frames_tv = min(mapper.num_frames, max_frames) if max_frames else mapper.num_frames

    # Directories for per-head TV outputs
    per_head_tv_dir = output_dir / "per_head_tv"
    per_head_tv_dir.mkdir(parents=True, exist_ok=True)
    tv_heat_dir = output_dir / "per_head_tv_heatmaps"
    tv_heat_dir.mkdir(parents=True, exist_ok=True)
    tv_token_dir = output_dir / "per_head_tv_top_tokens"
    tv_token_dir.mkdir(parents=True, exist_ok=True)

    # Token pieces aligned to text_idx order
    text_token_pieces = [processor.tokenizer.decode([int(t)]) for t in inputs.input_ids[0][text_idx]]

    tv_scores = []  # Collect summary scores per head/frame

    for frame_idx in range(actual_num_frames_tv):
        frame_vision_idx = mapper.get_frame_indices(frame_idx)
        for head_idx in range(num_heads_tv):
            A = attn_probs[0, head_idx, :, :]  # [Q, K]
            # Slice Text→Vision (rows=text tokens, cols=vision of this frame)
            attn_tv_head = torch.index_select(A, 0, text_idx)
            attn_tv_head = torch.index_select(attn_tv_head, 1, frame_vision_idx)  # [T_text, V_merged]

            # Save raw per-head TV matrix (merged grid)
            tv_np = attn_tv_head.detach().float().cpu().numpy()
            np.save(per_head_tv_dir / f"frame_{frame_idx:03d}_head{head_idx:02d}_tv.npy", tv_np)

            # Scores
            overall_score = float(tv_np.mean())
            any_high_score = float(tv_np.max())
            tv_scores.append({
                "frame_idx": frame_idx,
                "head_idx": head_idx,
                "overall_score": overall_score,
                "any_high_score": any_high_score,
            })

            # Top tokens by contribution (sum over vision patches)
            token_weights = tv_np.sum(axis=1)  # [T_text]
            order = np.argsort(token_weights)[::-1]
            k = min(len(order), 10)
            top_list = []
            for r in order[:k]:
                top_list.append({
                    "abs_index": int(text_idx[r]),
                    "piece": text_token_pieces[r],
                    "weight": float(token_weights[r])
                })
            with open(tv_token_dir / f"frame_{frame_idx:03d}_head{head_idx:02d}_top_tokens.json", "w") as f:
                json.dump({
                    "frame_idx": frame_idx,
                    "head_idx": head_idx,
                    "top_tokens": top_list
                }, f, indent=2)

            # Heatmap: average across text tokens to spatial map, upsample to pre-merger grid
            v_mean = attn_tv_head.mean(dim=0)  # [V_merged]
            nv_merged = mapper.patches_per_frame
            if nv_merged != pre_patches:
                ratio = pre_patches // nv_merged
                merger_side = int(round(math.sqrt(ratio)))
                post_h, post_w = mapper.spatial_grid
                v_hw = v_mean.view(post_h, post_w)
                v_hw = torch.repeat_interleave(v_hw, repeats=merger_side, dim=0)
                v_hw = torch.repeat_interleave(v_hw, repeats=merger_side, dim=1)
                v_pre = v_hw.view(pre_patches)
            else:
                v_pre = v_mean

            # Build a minimal [T, V_pre] for the plotting util (T=1)
            tv_for_plot = v_pre.unsqueeze(0).detach().float().cpu().numpy()
            viz = AttentionVisualizer(str(tv_heat_dir))
            viz.plot_frame_to_text_attention(
                attention=tv_for_plot,
                frame_idx=frame_idx,
                spatial_grid=(pre_h, pre_w),
                text_tokens=["<mean>"]
            )

    # Save per-head TV summary CSV
    import csv
    with open(output_dir / "per_head_tv_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "head_idx", "overall_score", "any_high_score"])
        writer.writeheader()
        for row in tv_scores:
            writer.writerow(row)

    # Vision→Text Per-Head Extraction (No Averaging)
    frame_attentions_vt_per_head = []
    if vision_to_text:
        print(f"\nExtracting Vision→Text per-head (no averaging)")

        # Get attention shape info
        num_heads = attn_probs.shape[1]

        # Apply limits for memory safety
        actual_num_frames = min(mapper.num_frames, max_frames) if max_frames else mapper.num_frames

        # Limit text tokens if specified
        if max_q_tokens and len(text_idx) > max_q_tokens:
            text_idx_limited = text_idx[:max_q_tokens]
            print(f"Limiting text tokens: {len(text_idx)} -> {max_q_tokens}")
        else:
            text_idx_limited = text_idx

        print(f"Extracting {actual_num_frames} frames x {num_heads} heads")
        print(f"Shape per head: [{pre_patches}, {len(text_idx_limited)}]")

        for frame_idx in range(actual_num_frames):
            frame_vision_idx = mapper.get_frame_indices(frame_idx)
            per_head_attns = []

            for head_idx in range(num_heads):
                # Extract Vision→Text for this specific head: [V, T]
                # attn_probs shape: [B, H, Q, K] where B=1
                # We want: vision patches (rows) × text tokens (cols)
                frame_attn_vt_head = attn_probs[0, head_idx, :, :]  # [Q, K]

                # Slice: vision rows, text cols
                frame_attn_vt_head = torch.index_select(frame_attn_vt_head, 0, frame_vision_idx)  # [V_merged, K]
                frame_attn_vt_head = torch.index_select(frame_attn_vt_head, 1, text_idx_limited)   # [V_merged, T]

                # Normalize across text tokens (each vision patch sums to 1 over text)
                frame_attn_vt_head = torch.nn.functional.normalize(frame_attn_vt_head, p=1, dim=1)

                # Upsample if merger was applied
                nv_merged = mapper.patches_per_frame
                if nv_merged != pre_patches:
                    ratio = pre_patches // nv_merged
                    merger_side = int(round(math.sqrt(ratio)))

                    # Upsample VISION dimension (rows)
                    post_h, post_w = mapper.spatial_grid
                    attn_hw_vt = frame_attn_vt_head.view(post_h, post_w, frame_attn_vt_head.size(1))
                    attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=0)
                    attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=1)
                    frame_attn_vt_head = attn_hw_vt.view(pre_patches, frame_attn_vt_head.size(1))

                # Convert to numpy: [V, T]
                attn_vt_head_np = frame_attn_vt_head.detach().float().cpu().numpy()
                per_head_attns.append(attn_vt_head_np)

            frame_attentions_vt_per_head.append(per_head_attns)

        print(f"Extracted {actual_num_frames} frames x {num_heads} heads")

        # Optional: Visualize per-head attention
        if visualize_per_head:
            print(f"\nVisualizing per-head attention")
            visualizer = AttentionVisualizer(output_dir)
            heads_to_vis = min(num_heads, max_vis_heads) if max_vis_heads else num_heads

            viz_count = 0
            for frame_idx, per_head_attns in enumerate(frame_attentions_vt_per_head):
                for head_idx in range(heads_to_vis):
                    visualizer.plot_vision_to_text_per_head(
                        attention_head=per_head_attns[head_idx],
                        frame_idx=frame_idx,
                        head_idx=head_idx,
                        text_tokens=question_tokens,
                        spatial_grid=(pre_h, pre_w),
                        normalize_rows=normalize_rows
                    )
                    viz_count += 1

            limit_msg = f" (limited to {max_vis_heads}/{num_heads} heads)" if max_vis_heads and num_heads > max_vis_heads else ""
            print(f"Saved {viz_count} visualizations{limit_msg}")

        # Optional: Save top-k token analysis
        if save_top_k_analysis:
            print(f"\nAnalyzing top-{top_k} tokens per patch")
            analysis_dir = output_dir / "top_k_analysis"
            analysis_dir.mkdir(exist_ok=True)

            for frame_idx, per_head_attns in enumerate(frame_attentions_vt_per_head):
                for head_idx, attn_head in enumerate(per_head_attns):
                    top_k_results = analyze_top_k_tokens(attn_head, question_tokens, k=top_k)
                    analysis_file = analysis_dir / f"frame_{frame_idx:03d}_head{head_idx:02d}_topk.json"

                    with open(analysis_file, 'w') as f:
                        json.dump({
                            'frame_idx': frame_idx,
                            'head_idx': head_idx,
                            'top_k': top_k,
                            'per_patch_analysis': top_k_results
                        }, f, indent=2)

            total_files = actual_num_frames * num_heads
            print(f"Saved {total_files} top-k analysis files")

    # Save tensors
    print(f"\nSaving tensors")
    torch.save(attn_probs, output_dir / "prefill_attention.pt")

    for i in range(mapper.num_frames):
        np.save(output_dir / f"frame_{i:03d}_text_to_vision.npy", frame_attentions_tv[i])
        np.save(output_dir / f"frame_{i:03d}_vision_to_text.npy", frame_attentions_vt[i])

    print(f"Saved {mapper.num_frames} x 2 averaged attention matrices")

    # Save Vision→Text per-head tensors
    if vision_to_text and len(frame_attentions_vt_per_head) > 0:
        num_heads = len(frame_attentions_vt_per_head[0])
        num_frames_saved = len(frame_attentions_vt_per_head)
        text_tokens_used = len(text_idx_limited) if max_q_tokens and len(text_idx) > max_q_tokens else len(text_idx)

        total_files_saved = 0
        for frame_idx, per_head_attns in enumerate(frame_attentions_vt_per_head):
            for head_idx, attn_head_np in enumerate(per_head_attns):
                filename = output_dir / f"frame_{frame_idx:03d}_vision_to_text_head{head_idx:02d}.npy"
                np.save(filename, attn_head_np)
                total_files_saved += 1

        print(f"Saved {total_files_saved} per-head tensors ({num_frames_saved} frames x {num_heads} heads)")
        print(f"Shape per file: [{pre_patches}, {text_tokens_used}]")

    # Save decode attentions
    if capture_decode and len(capture.decode_attentions) > 0:
        for i, attn in enumerate(capture.decode_attentions):
            torch.save(attn, output_dir / f"decode_step_{i:02d}_attention.pt")
        print(f"Saved {len(capture.decode_attentions)} decode-step tensors")

    # Create visualizations
    print(f"\nCreating visualizations")

    visualizer = AttentionVisualizer(output_dir)

    for frame_idx in range(mapper.num_frames):
        # TEXT→VISION
        visualizer.plot_frame_to_text_attention(
            frame_attentions_tv[frame_idx],
            frame_idx,
            (pre_h, pre_w),
            question_tokens,
            save_path=output_dir / f"frame_{frame_idx:03d}_text_to_vision.png"
        )

        # VISION→TEXT (raw matrix, no averaging)
        visualizer.plot_matrix_no_avg(
            frame_attentions_vt[frame_idx],
            title=f"Frame {frame_idx}: Vision→Text Attention (Layer 0, prefill)",
            x_label="Text Token Index",
            y_label="Vision Patch Index",
            save_path=output_dir / f"frame_{frame_idx:03d}_vision_to_text.png"
        )

    print(f"Saved {mapper.num_frames} x 2 visualization types")

    # Summary visualizations
    visualizer.plot_all_frames_summary(frame_attentions_tv, (pre_h, pre_w))
    print(f"Saved summary plots")

    # Save metadata
    metadata = {
        "model_path": model_path,
        "model_base": model_base,
        "video_path": video_path,
        "question": question,
        "num_frames": mapper.num_frames,
        "spatial_grid": mapper.spatial_grid,
        "pre_merger_grid": mapper.pre_merger_grid,
        "patches_per_frame": mapper.patches_per_frame,
        "num_text_tokens": len(text_idx),
        "attention_shape": list(attn_probs.shape),
        "generated_text": generated_text[0],
        "vision_to_text_enabled": vision_to_text,
    }

    if vision_to_text and len(frame_attentions_vt_per_head) > 0:
        metadata["vision_to_text_num_heads"] = len(frame_attentions_vt_per_head[0])
        metadata["vision_to_text_num_frames"] = len(frame_attentions_vt_per_head)
        metadata["vision_to_text_text_tokens"] = len(text_idx_limited) if max_q_tokens and len(text_idx) > max_q_tokens else len(text_idx)
        metadata["vision_to_text_vision_patches"] = pre_patches

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE")
    print(f"=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Frames: {mapper.num_frames}")
    print(f"TEXT→VISION attention range: [{min(a.mean() for a in frame_attentions_tv):.6f}, {max(a.mean() for a in frame_attentions_tv):.6f}]")
    print(f"VISION→TEXT attention range: [{min(a.mean() for a in frame_attentions_vt):.6f}, {max(a.mean() for a in frame_attentions_vt):.6f}]")

    if vision_to_text and len(frame_attentions_vt_per_head) > 0:
        num_heads = len(frame_attentions_vt_per_head[0])
        num_frames_saved = len(frame_attentions_vt_per_head)
        print(f"Vision→Text per-head: {num_frames_saved} frames x {num_heads} heads ({num_frames_saved * num_heads} files)")

    result = {
        'attention': attn_probs,
        'frame_attentions_text_to_vision': frame_attentions_tv,
        'frame_attentions_vision_to_text': frame_attentions_vt,
        'mapper': mapper,
        'metadata': metadata,
        'generated_text': generated_text
    }

    if vision_to_text and len(frame_attentions_vt_per_head) > 0:
        result['frame_attentions_vision_to_text_per_head'] = frame_attentions_vt_per_head

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract attention maps using StreamingVLM")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, required=True, help="Question about the video")
    parser.add_argument("--output_dir", type=str, default="outputs/attn", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS for preprocessing the video")
    parser.add_argument("--model_path", type=str, default="mit-han-lab/StreamingVLM", help="Model path")
    parser.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2"], default="Qwen2_5", help="Base model type")
    parser.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"], help="Position mode")
    parser.add_argument("--all_text", action="store_true", default=False, help="All PEs are 1D")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--capture_decode", action="store_true", help="Capture decode-step attention")
    parser.add_argument("--max_decode_steps", type=int, default=4, help="Max decode steps to capture")

    # Vision→Text per-head extraction arguments
    parser.add_argument("--vision_to_text", action="store_true", help="Extract Vision→Text attention per-head (no averaging)")
    parser.add_argument("--max_q_tokens", type=int, default=None, help="Limit text tokens for memory safety (default: no limit)")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for memory safety (default: no limit)")

    # Enhanced analysis arguments
    parser.add_argument("--filter_video_pad", action="store_true", help="Filter out <|video_pad|> tokens from text analysis")
    parser.add_argument("--visualize_per_head", action="store_true", help="Generate PNG visualizations for each head")
    parser.add_argument("--max_vis_heads", type=int, default=None, help="Limit number of heads to visualize (default: all)")
    parser.add_argument("--save_top_k_analysis", action="store_true", help="Save JSON files with top-k attended tokens per patch")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to analyze per patch (default: 5)")
    parser.add_argument("--no_normalize_rows", action="store_true", help="Disable row normalization in per-head visualization")

    args = parser.parse_args()

    extract_attention_streaming(
        model_path=args.model_path,
        model_base=args.model_base,
        pos_mode=args.pos_mode,
        all_text=args.all_text,
        video_path=args.video_path,
        question=args.question,
        output_dir=args.output_dir,
        fps=args.fps,
        device=args.device,
        capture_decode=args.capture_decode,
        max_decode_steps=args.max_decode_steps,
        vision_to_text=args.vision_to_text,
        max_q_tokens=args.max_q_tokens,
        max_frames=args.max_frames,
        filter_video_pad=args.filter_video_pad,
        visualize_per_head=args.visualize_per_head,
        max_vis_heads=args.max_vis_heads,
        save_top_k_analysis=args.save_top_k_analysis,
        top_k=args.top_k,
        normalize_rows=not args.no_normalize_rows
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())
