#!/usr/bin/env python3
"""
Qwen3-VL first-layer Q·Kᵀ extraction with chunked vision tokens.

This script runs a single prefill pass (FlashAttention stays enabled) and
captures the first transformer block's q/k projections to compute vision→text
attention logits without instantiating full attention matrices.

CLI example:
    python extract_qk_scores_qwen3vl.py \
        --model_path Qwen/Qwen3-VL-4B-Instruct \
        --video_path demo.mp4 \
        --question "What is happening?" \
        --output_dir outputs/qk_scores_demo \
        --chunk_size 4096 \
        --apply_softmax true
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("transformers is required for this script") from exc

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False


DEFAULT_IMAGE_PATCH_SIZE = 16
DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in {"true", "1", "yes", "y"}:
        return True
    if v_lower in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{v}'")


def load_model_and_processor(
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: Optional[str],
) -> Tuple[nn.Module, AutoProcessor]:
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(DTYPE_MAP.keys())}.")

    attn_kwargs = {}
    if attn_implementation:
        attn_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=DTYPE_MAP[dtype],
        device_map=None,
        **attn_kwargs,
    )
    target_device = torch.device(device)
    if device != "auto":
        model.to(target_device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor


def _resolve_layers_container(model: nn.Module) -> List[nn.Module]:
    candidate_paths = [
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("model", "transformer", "layers"),
        ("transformer", "layers"),
        ("layers",),
    ]
    for path in candidate_paths:
        node = model
        valid = True
        for attr in path:
            if not hasattr(node, attr):
                valid = False
                break
            node = getattr(node, attr)
        if not valid:
            continue
        try:
            _ = len(node)
        except TypeError:
            continue
        if len(node) == 0:
            continue
        first = node[0]
        if any(hasattr(first, cand) for cand in ("self_attn", "self_attention", "attention")):
            return node
    raise RuntimeError("Unable to locate decoder layers with a self-attention module.")


def _get_first_attn_module(model: nn.Module) -> nn.Module:
    layers = _resolve_layers_container(model)
    first_block = layers[0]
    for attr in ("self_attn", "self_attention", "attention"):
        if hasattr(first_block, attr):
            return getattr(first_block, attr)
    raise RuntimeError("First decoder block has no recognizable attention module.")


class QKCapture:
    """Hook helper to capture q/k projection outputs."""

    def __init__(self):
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, which: str):
        def _inner(_module, _inp, output):
            if getattr(self, which) is None:
                setattr(self, which, output.detach())
        return _inner

    def register(self, attn_module: nn.Module):
        if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "k_proj"):
            raise RuntimeError("Attention module missing q_proj or k_proj; cannot capture.")
        self.handles.append(attn_module.q_proj.register_forward_hook(self._hook("q")))
        self.handles.append(attn_module.k_proj.register_forward_hook(self._hook("k")))

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class TokenIndexMapper:
    """Reuse tokenizer-aware logic to locate vision and question token spans."""

    def __init__(self, input_ids: torch.Tensor, video_token_id: int, video_grid_thw: torch.Tensor):
        self.input_ids = input_ids
        self.video_token_id = video_token_id
        self.video_grid_thw = video_grid_thw
        self.device = input_ids.device

        video_mask = (input_ids[0] == video_token_id)
        self.vision_idx = torch.where(video_mask)[0].to(self.device)
        if self.vision_idx.numel() == 0:
            raise RuntimeError("No vision tokens detected; check tokenizer special tokens.")

        if len(video_grid_thw.shape) == 2:
            t, h, w = video_grid_thw[0]
        else:
            t, h, w = video_grid_thw
        self.num_frames = int(t.item())
        self.pre_merger_grid = (int(h.item()), int(w.item()))
        total_vision_tokens = int(self.vision_idx.numel())
        self.patches_per_frame = max(1, total_vision_tokens // max(1, self.num_frames))

    def get_question_tokens(
        self,
        processor,
        filter_special: bool = True,
        filter_video_pad: bool = True,
    ) -> torch.Tensor:
        ids = self.input_ids[0]
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = processor.tokenizer.convert_tokens_to_ids("user")
        nl_id = processor.tokenizer.convert_tokens_to_ids("\n")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

        text_start, text_end = None, None
        for i in range(len(ids)):
            if ids[i].item() == im_start_id:
                end = i + 1
                while end < len(ids) and ids[end].item() != im_end_id:
                    end += 1
                if end < len(ids) and ids[i + 1].item() == user_id:
                    start = i + 2
                    if start < len(ids) and ids[start].item() == nl_id:
                        start += 1
                    text_start, text_end = start, end
                    break
        if text_start is None:
            text_start = int(self.vision_idx[-1].item()) + 1
            text_end = len(ids)

        text_idx = torch.arange(text_start, text_end, device=self.device)

        if filter_special:
            special_ids = set(
                processor.tokenizer.convert_tokens_to_ids(tok)
                for tok in ("<|im_start|>", "<|im_end|>", "assistant", "<|endoftext|>")
                if processor.tokenizer.convert_tokens_to_ids(tok) is not None
            )
            mask = torch.ones_like(text_idx, dtype=torch.bool)
            for j, idx in enumerate(text_idx):
                if int(ids[idx]) in special_ids:
                    mask[j] = False
            text_idx = text_idx[mask]
        if filter_video_pad:
            mask = torch.ones_like(text_idx, dtype=torch.bool)
            for j, idx in enumerate(text_idx):
                if int(ids[idx]) == video_pad_id:
                    mask[j] = False
            text_idx = text_idx[mask]
        if text_idx.numel() == 0:
            raise RuntimeError("Text token range is empty after filtering.")
        return text_idx


def prepare_inputs(
    video_path: str,
    question: str,
    processor: AutoProcessor,
    device: torch.device,
    fps: float,
    max_frames: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[List]]]:
    if not HAS_QWEN_VL_UTILS:
        raise ImportError("qwen_vl_utils is required for video preprocessing.")

    video_payload: Dict = {"type": "video", "video": video_path}
    if fps is not None:
        video_payload["fps"] = fps
    if max_frames is not None:
        video_payload["max_frames"] = max_frames

    messages = [
        {
            "role": "user",
            "content": [
                video_payload,
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs = None
    video_inputs = None
    video_kwargs = {}
    video_metadatas = None
    process_kwargs = dict(
        image_patch_size=DEFAULT_IMAGE_PATCH_SIZE,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    try:
        image_inputs, raw_video_inputs, video_kwargs = process_vision_info(messages, **process_kwargs)
    except TypeError as exc:
        print(
            f"Warning: process_vision_info missing metadata kwargs ({exc}); falling back without metadata capture."
        )
        fallback_kwargs = dict(
            image_patch_size=DEFAULT_IMAGE_PATCH_SIZE,
            return_video_kwargs=True,
        )
        image_inputs, raw_video_inputs, video_kwargs = process_vision_info(messages, **fallback_kwargs)
        video_metadatas = None
    else:
        if raw_video_inputs is not None and len(raw_video_inputs) > 0 and isinstance(raw_video_inputs[0], tuple):
            videos, metas = zip(*raw_video_inputs)
            video_inputs = list(videos)
            video_metadatas = [dict(metadata) for metadata in metas]
        else:
            video_inputs = raw_video_inputs
    if video_metadatas is None and raw_video_inputs is not None and video_inputs is None:
        video_inputs = raw_video_inputs

    processor_kwargs = dict(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )
    if video_metadatas is not None:
        processor_kwargs["video_metadata"] = video_metadatas
    if video_kwargs:
        processor_kwargs.update(video_kwargs)

    inputs = processor(**processor_kwargs)
    inputs = inputs.to(device)
    media_meta = {
        "video_metadatas": video_metadatas,
        "video_kwargs": video_kwargs,
        "text_prompt": text_prompt,
    }
    return inputs, media_meta


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    bsz, num_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, num_repeats, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_heads * num_repeats, seq_len, head_dim)


def reshape_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_module: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if q.dim() != 3 or k.dim() != 3:
        raise RuntimeError(f"Unexpected q/k shape: {q.shape}, {k.shape}")
    bsz, qlen, hidden = q.shape
    kval_hidden = k.shape[-1]

    def _maybe_int(source, attr_names):
        if source is None:
            return None
        for name in attr_names:
            if not hasattr(source, name):
                continue
            value = getattr(source, name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _linear_out_features(linear: Optional[nn.Module]) -> Optional[int]:
        if linear is None:
            return None
        if hasattr(linear, "out_features"):
            return int(linear.out_features)
        weight = getattr(linear, "weight", None)
        if weight is not None:
            return int(weight.shape[0])
        return None

    config = getattr(attn_module, "config", None)
    num_heads = _maybe_int(attn_module, ("num_heads", "num_attention_heads"))
    if num_heads is None:
        num_heads = _maybe_int(config, ("num_attention_heads", "num_heads"))

    head_dim = _maybe_int(attn_module, ("head_dim",))
    if head_dim is None:
        head_dim = _maybe_int(config, ("head_dim",))

    if num_heads is None and head_dim is None:
        raise RuntimeError("Unable to infer attention head metadata (num_heads/head_dim).")

    q_proj_out = _linear_out_features(getattr(attn_module, "q_proj", None)) or hidden
    k_proj_out = _linear_out_features(getattr(attn_module, "k_proj", None)) or kval_hidden

    if num_heads is None and head_dim is not None and q_proj_out % head_dim == 0:
        num_heads = q_proj_out // head_dim
    if head_dim is None and num_heads is not None and q_proj_out % num_heads == 0:
        head_dim = q_proj_out // num_heads
    if head_dim is None and num_heads is not None:
        head_dim = hidden // num_heads
    if num_heads is None and head_dim is not None:
        num_heads = hidden // head_dim

    num_kv_heads = _maybe_int(attn_module, ("num_key_value_heads", "num_kv"))
    if num_kv_heads is None:
        num_kv_heads = _maybe_int(config, ("num_key_value_heads", "num_kv_heads", "num_kv"))

    if num_kv_heads is None:
        num_kv_groups = _maybe_int(attn_module, ("num_key_value_groups",))
        if num_kv_groups is None:
            num_kv_groups = _maybe_int(config, ("num_key_value_groups",))
        if num_kv_groups and num_heads:
            num_kv_heads = max(1, num_heads // num_kv_groups)

    if num_kv_heads is None and head_dim is not None and k_proj_out % head_dim == 0:
        num_kv_heads = k_proj_out // head_dim

    if num_kv_heads is None:
        num_kv_heads = num_heads

    if num_heads is None or head_dim is None:
        raise RuntimeError("Attention module missing resolved num_heads/head_dim attributes.")

    q = q.view(bsz, qlen, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, qlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    if num_kv_heads != num_heads:
        if num_heads % num_kv_heads != 0:
            raise RuntimeError("num_heads must be divisible by num_key_value_heads for repetition.")
        repeat_factor = num_heads // num_kv_heads
        k = repeat_kv(k, repeat_factor)
    return q, k, num_heads, head_dim


def tensor_to_list(tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return [int(x) for x in tensor.reshape(-1).tolist()]
    return tensor


def compute_temporal_mapping(
    num_temporal_tokens: int,
    sampled_indices: Optional[List[int]],
    temporal_patch_size: Optional[int],
) -> Dict[str, List[int]]:
    if sampled_indices is None:
        return {str(i): [] for i in range(num_temporal_tokens)}
    if temporal_patch_size is None or temporal_patch_size <= 0:
        if num_temporal_tokens > 0:
            temporal_patch_size = math.ceil(len(sampled_indices) / num_temporal_tokens)
        else:
            temporal_patch_size = len(sampled_indices)
    mapping: Dict[str, List[int]] = {}
    for t in range(num_temporal_tokens):
        start = t * temporal_patch_size
        end = start + temporal_patch_size
        mapping[str(t)] = [int(x) for x in sampled_indices[start:end] if start < len(sampled_indices)]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first-layer Q·Kᵀ scores from Qwen3-VL prefill.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local checkpoint.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (e.g., cuda:0 or cpu).")
    parser.add_argument("--dtype", type=str, default="auto", choices=list(DTYPE_MAP.keys()), help="Model dtype.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Passed to from_pretrained to keep FlashAttention on.",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Client-side sampling FPS hint.")
    parser.add_argument("--max_frames", type=int, default=32, help="Optional max frames for process_vision_info.")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Vision token chunk size for matmul.")
    parser.add_argument("--apply_softmax", type=str2bool, default=False, help="Apply softmax over text tokens.")
    parser.add_argument("--top_k", type=int, default=5, help="How many temporal tokens to print.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen3-VL Q·Kᵀ extraction")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Video: {args.video_path}")
    print(f"Question: {args.question}")
    print(f"Output dir: {output_dir}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=device.type if args.device == "auto" else args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    target_device = next(model.parameters()).device
    print(f"Model device: {target_device} | dtype: {args.dtype}")

    inputs, media_meta = prepare_inputs(
        video_path=args.video_path,
        question=args.question,
        processor=processor,
        device=target_device,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"Input ids shape: {tuple(inputs.input_ids.shape)}")
    video_grid_thw = inputs.video_grid_thw
    print(f"video_grid_thw: {video_grid_thw}")

    attn_module = _get_first_attn_module(model)
    capture = QKCapture()
    capture.register(attn_module)

    with torch.no_grad():
        _ = model(**inputs, use_cache=True)
    capture.clear()

    if capture.q is None or capture.k is None:
        raise RuntimeError("Failed to capture q/k projections. Check hook wiring.")

    q, k, num_heads, head_dim = reshape_qk(capture.q, capture.k, attn_module)

    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if video_token_id is None:
        raise RuntimeError("Tokenizer does not contain <|video_pad|> token.")
    mapper = TokenIndexMapper(inputs.input_ids, video_token_id, video_grid_thw)
    vision_indices = mapper.vision_idx
    thw = video_grid_thw[0] if len(video_grid_thw.shape) == 2 else video_grid_thw
    T, H, W = (int(thw[0]), int(thw[1]), int(thw[2]))
    if T <= 0:
        raise RuntimeError(f"Invalid temporal dimension extracted from video_grid_thw: {T}.")
    num_vis_expected = T * H * W
    num_vis_found = int(vision_indices.numel())
    if num_vis_found == 0:
        raise RuntimeError("No vision tokens detected via <|video_pad|>; cannot continue.")

    spatial_tokens_per_frame: Optional[int] = None
    if num_vis_found < num_vis_expected:
        if num_vis_found % T != 0:
            raise RuntimeError(
                f"Vision token count {num_vis_found} not divisible by T={T}; cannot infer per-frame spans."
            )
        spatial_tokens_per_frame = num_vis_found // T
        merge_factor = (H * W) / float(spatial_tokens_per_frame)
        print(
            "Warning: "
            f"Found {num_vis_found} vision tokens but THW expects {num_vis_expected}. "
            f"Assuming {spatial_tokens_per_frame} spatial tokens/frame (merge factor ≈ {merge_factor:.2f})."
        )
    else:
        if num_vis_found > num_vis_expected:
            print(
                f"Warning: Found {num_vis_found} vision tokens but THW expects {num_vis_expected}; truncating to expected span."
            )
        vision_indices = vision_indices[:num_vis_expected]
        num_vis_found = int(vision_indices.numel())
        spatial_tokens_per_frame = H * W

    spatial_tokens_per_frame = max(1, spatial_tokens_per_frame)
    spatial_merge_factor = float(H * W) / float(spatial_tokens_per_frame)
    num_vis = num_vis_found

    text_indices = mapper.get_question_tokens(processor, filter_special=True, filter_video_pad=True)
    text_len = int(text_indices.numel())

    print(f"M (vision tokens) = {num_vis}, N (text tokens) = {text_len}, num_heads = {num_heads}, head_dim = {head_dim}")
    print(f"apply_softmax={args.apply_softmax}")

    q_vis = q.index_select(dim=2, index=vision_indices.to(q.device))
    k_txt = k.index_select(dim=2, index=text_indices.to(k.device))
    compute_dtype = torch.float32 if q_vis.dtype in (torch.float16, torch.bfloat16) else q_vis.dtype
    k_txt_t = k_txt.transpose(-1, -2).to(compute_dtype)

    frame_scores_sum = torch.zeros(
        (q_vis.shape[0], num_heads, T, text_len),
        dtype=compute_dtype,
        device=q_vis.device,
    )
    scale = 1.0 / math.sqrt(head_dim)
    text_tokens = [processor.tokenizer.decode([int(inputs.input_ids[0, idx])]).strip() for idx in text_indices]

    indices_template = torch.arange(num_vis, device=q_vis.device, dtype=torch.long)

    for start in range(0, num_vis, args.chunk_size):
        end = min(num_vis, start + args.chunk_size)
        q_chunk = q_vis[:, :, start:end, :].to(compute_dtype)
        chunk_scores = torch.matmul(q_chunk, k_txt_t)
        chunk_scores = chunk_scores * scale
        if args.apply_softmax:
            chunk_scores = torch.softmax(chunk_scores, dim=-1)

        frame_idx = indices_template[start:end] // spatial_tokens_per_frame
        frame_idx = frame_idx.clamp(max=T - 1)
        scatter_index = frame_idx.view(1, -1, 1).expand(q_vis.shape[0] * num_heads, -1, text_len)
        dest = frame_scores_sum.view(q_vis.shape[0] * num_heads, T, text_len)
        src = chunk_scores.reshape(q_vis.shape[0] * num_heads, end - start, text_len)
        dest.scatter_add_(dim=1, index=scatter_index, src=src)

    frame_scores_mean = frame_scores_sum / float(max(1, spatial_tokens_per_frame))
    per_head_text_mean = frame_scores_mean.mean(dim=-1)  # [B, H, T]
    per_frame_scalar = per_head_text_mean.mean(dim=1)  # [B, T]

    top_k = min(args.top_k, T)
    top_scores, top_indices = torch.topk(per_frame_scalar[0], k=top_k)
    print("Top temporal slots by aggregated score:")
    for rank in range(top_k):
        print(f"  #{rank+1}: T={int(top_indices[rank])} | score={float(top_scores[rank]):.4f}")

    temporal_patch_size = getattr(getattr(model.config, "vision_config", None), "temporal_patch_size", None)
    if temporal_patch_size is not None:
        temporal_patch_size = int(temporal_patch_size)
    sampled_indices = None
    video_metadatas = media_meta.get("video_metadatas")
    if video_metadatas:
        meta_frames = video_metadatas[0].get("frames_indices")
        sampled_indices = tensor_to_list(meta_frames)
    temporal_mapping = compute_temporal_mapping(T, sampled_indices, temporal_patch_size)

    raw_scores_path = output_dir / "qk_scores_raw.pt"
    torch.save(frame_scores_mean.to("cpu"), raw_scores_path)
    print(f"Saved per-head per-text frame scores to {raw_scores_path}")

    frame_scores_payload = {
        "per_head_text_mean": per_head_text_mean.cpu().tolist(),
        "per_frame_mean": per_frame_scalar.cpu().tolist(),
        "text_tokens": text_tokens,
        "apply_softmax": bool(args.apply_softmax),
    }
    (output_dir / "frame_scores.json").write_text(json.dumps(frame_scores_payload, indent=2))

    meta = {
        "model_path": args.model_path,
        "video_path": args.video_path,
        "question": args.question,
        "video_grid_thw": [int(T), int(H), int(W)],
        "num_vision_tokens": num_vis,
        "num_vision_tokens_expected": num_vis_expected,
        "num_text_tokens": text_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "chunk_size": args.chunk_size,
        "apply_softmax": bool(args.apply_softmax),
        "spatial_tokens_per_frame": int(spatial_tokens_per_frame),
        "spatial_merge_factor": spatial_merge_factor,
        "temporal_patch_size": temporal_patch_size,
        "temporal_mapping": temporal_mapping,
        "text_tokens": text_tokens,
        "sampled_frame_indices": sampled_indices,
        "processor_text_prompt": media_meta.get("text_prompt"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata to {output_dir / 'meta.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
