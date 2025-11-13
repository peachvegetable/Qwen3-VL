#!/usr/bin/env python3
"""
Streaming Q·Kᵀ attention reconstruction for Qwen3-VL layer-0 prefill.

The script mirrors the reducer / visualiser pipeline used in
`extract_attention_qwen3vl.py`, but instead of reading attention weights from
FlashAttention it recomputes them chunk-by-chunk from cached hidden states and
never instantiates the full [B, H, Q, Q] tensor. Only the text-side keys are
kept resident; vision rows are streamed, immediately pooled, and (optionally)
written to disk for fixed-range [0,1] visualisations.
"""

from __future__ import annotations

import argparse
import json
import math
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:  # pragma: no cover - optional
    HAS_SEABORN = False

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:  # pragma: no cover - optional
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None  # type: ignore

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from extract_attention_qwen3vl import (
    DEFAULT_IMAGE_PATCH_SIZE,
    TokenIndexMapper,
    load_model_and_processor,
    _resolve_layers_container,
)


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def _is_punctuation_like(token: str) -> bool:
    stripped = token.replace("Ġ", "").replace("▁", "").strip()
    if not stripped:
        return True
    if any(ch.isalnum() for ch in stripped):
        return False
    return all(ch in string.punctuation for ch in stripped)


def build_content_mask(
    tokenizer,
    token_ids: List[int],
    decoded_tokens: List[str],
    question: str,
) -> List[bool]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    extra_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|image_start|>",
        "<|image_end|>",
        "<|video_pad|>",
        "<|video_start|>",
        "<|video_end|>",
        "<|endoftext|>",
        "assistant",
        "user",
        "system",
        "\n",
    ]
    for tok in extra_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None:
            continue
        special_ids.add(int(tok_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is not None:
            special_ids.add(int(tok_id))

    base_mask: List[bool] = []
    for token_id, token_str in zip(token_ids, decoded_tokens):
        is_content = True
        if int(token_id) in special_ids:
            is_content = False
        elif _is_punctuation_like(token_str):
            is_content = False
        base_mask.append(is_content)

    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    if not question_ids:
        return base_mask
    for start in range(len(token_ids) - len(question_ids) + 1):
        if token_ids[start : start + len(question_ids)] == question_ids:
            for pos in range(start, start + len(question_ids)):
                if 0 <= pos < len(base_mask):
                    base_mask[pos] = True
            break
    return base_mask


def _find_subsequence(sequence: List[int], pattern: List[int]) -> Optional[int]:
    if not pattern or len(pattern) > len(sequence):
        return None
    plen = len(pattern)
    for start in range(len(sequence) - plen + 1):
        if sequence[start : start + plen] == pattern:
            return start
    return None


def prepare_inputs(
    processor,
    video_path: str,
    question: str,
    device: torch.device,
    fps: float,
    max_frames: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[List]]]:
    if not HAS_QWEN_VL_UTILS:
        raise ImportError("qwen_vl_utils is required for video preprocessing")

    video_payload: Dict = {"type": "video", "video": video_path, "fps": fps}
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
    raw_video_inputs = None
    process_kwargs = dict(
        image_patch_size=DEFAULT_IMAGE_PATCH_SIZE,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    try:
        image_inputs, raw_video_inputs, video_kwargs = process_vision_info(messages, **process_kwargs)  # type: ignore[arg-type]
    except TypeError as exc:  # fallback when utils miss metadata kwargs
        print(
            f"Warning: process_vision_info missing metadata kwargs ({exc}); falling back without metadata capture."
        )
        fallback_kwargs = dict(image_patch_size=DEFAULT_IMAGE_PATCH_SIZE, return_video_kwargs=True)
        image_inputs, raw_video_inputs, video_kwargs = process_vision_info(messages, **fallback_kwargs)  # type: ignore[arg-type]
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


# ---------------------------------------------------------------------------
# Prefill capture
# ---------------------------------------------------------------------------

class PrefillStateCapture:
    def __init__(self):
        self.hidden_states: Optional[torch.Tensor] = None
        self.position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self._captured = False
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model, layer_idx: int = 0):
        layers = _resolve_layers_container(model)
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} exceeds number of layers {len(layers)}")
        target_module = layers[layer_idx].self_attn

        def _hook(_module, _args, kwargs, _output):
            if self._captured:
                return
            hidden_states = kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if hidden_states is None or position_embeddings is None:
                return
            self.hidden_states = hidden_states.detach()
            if isinstance(position_embeddings, (tuple, list)):
                self.position_embeddings = tuple(pe.detach() for pe in position_embeddings)
            else:
                self.position_embeddings = position_embeddings
            self.attention_mask = attention_mask.detach() if isinstance(attention_mask, torch.Tensor) else None
            self._captured = True

        handle = target_module.register_forward_hook(_hook, with_kwargs=True)
        self._handles.append(handle)

    def clear(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def pool_text_vector(row_vec: torch.Tensor, pool: str, topk: int) -> float:
    if row_vec.numel() == 0:
        return 0.0
    row_vec = row_vec.float()
    if pool == "mean" or topk <= 0:
        return float(row_vec.mean().item())
    if pool == "max":
        return float(row_vec.max().item())
    # top-k mean
    k = min(topk, row_vec.numel())
    return float(torch.topk(row_vec, k=k).values.mean().item())


def finalize_frame_scores(row_scores: List[List[float]], score_topk: int) -> List[float]:
    final_scores: List[float] = []
    for scores in row_scores:
        if not scores:
            final_scores.append(0.0)
            continue
        tensor = torch.tensor(scores, dtype=torch.float32)
        if score_topk > 0:
            k = min(score_topk, tensor.numel())
            final_scores.append(float(torch.topk(tensor, k=k).values.mean().item()))
        else:
            final_scores.append(float(tensor.mean().item()))
    return final_scores


def save_heatmap(matrix: np.ndarray, save_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    if HAS_SEABORN:
        sns.heatmap(matrix, cmap="hot", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(matrix, cmap="hot", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar()
    plt.title(title)
    plt.xlabel("Text token")
    plt.ylabel("Vision row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Streaming attention reconstruction
# ---------------------------------------------------------------------------

def stream_qk_attention(
    attn_module,
    capture: PrefillStateCapture,
    vision_indices: torch.Tensor,
    text_indices: torch.Tensor,
    mapper: TokenIndexMapper,
    content_idx_tensor: torch.Tensor,
    chunk_q: int,
    pool: str,
    pool_topk: int,
    score_topk: int,
    save_viz_mats: bool,
    no_head_avg: bool,
    output_dir: Path,
):
    if capture.hidden_states is None or capture.position_embeddings is None:
        raise RuntimeError("Prefill hidden states/positional embeddings were not captured.")

    hidden_states = capture.hidden_states
    cos, sin = capture.position_embeddings
    attention_mask = capture.attention_mask

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)

    query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
    query_states = attn_module.q_norm(query_states).transpose(1, 2)
    key_states = attn_module.k_proj(hidden_states).view(hidden_shape)
    key_states = attn_module.k_norm(key_states).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if attn_module.num_key_value_groups > 1:
        key_states = repeat_kv(key_states, attn_module.num_key_value_groups)

    batch, num_heads, seq_len, head_dim = query_states.shape
    assert batch == 1, "Only batch_size=1 is supported."

    device = query_states.device
    text_idx = text_indices.to(device)
    K_text = key_states.index_select(dim=2, index=text_idx).to(torch.float16)
    K_text_t = K_text.transpose(-1, -2)

    mask_for_text = None
    if attention_mask is not None:
        mask_for_text = attention_mask.index_select(-1, index=text_idx).to(torch.float32)

    num_frames = mapper.num_frames
    frame_row_scores: List[List[float]] = [[] for _ in range(num_frames)]

    viz_dir = output_dir / "viz_mats"
    if save_viz_mats:
        viz_dir.mkdir(parents=True, exist_ok=True)
        headavg_buffers: List[List[torch.Tensor]] = [[] for _ in range(num_frames)]
    else:
        headavg_buffers = []  # type: ignore

    total_rows = vision_indices.numel()
    print(
        f"Streaming attention: vision_rows={total_rows}, text_tokens={text_idx.numel()} via chunks of {chunk_q}."
    )

    for start in range(0, total_rows, chunk_q):
        end = min(total_rows, start + chunk_q)
        vision_pos = vision_indices[start:end].to(device)
        q_chunk = query_states.index_select(dim=2, index=vision_pos)
        scores = torch.matmul(q_chunk.to(torch.float16), K_text_t)
        scores = scores.to(torch.float32) * (1.0 / math.sqrt(head_dim))
        if mask_for_text is not None:
            mask_chunk = mask_for_text[:, :, vision_pos, :]
            scores = scores + mask_chunk
        probs = torch.softmax(scores, dim=-1)

        avg_chunk = probs.mean(dim=1)[0]  # [rows_chunk, N_text]
        if no_head_avg:
            per_head_chunk = probs[0]  # [H, rows_chunk, N_text]
        else:
            per_head_chunk = None

        row_ids = torch.arange(start, end, device=device)
        frame_ids = (row_ids // mapper.patches_per_frame).tolist()

        for local_row, frame in enumerate(frame_ids):
            if per_head_chunk is not None:
                for head in range(num_heads):
                    row_vec = per_head_chunk[head, local_row, content_idx_tensor]
                    row_score = pool_text_vector(row_vec, pool, pool_topk)
                    frame_row_scores[frame].append(row_score)
            else:
                row_vec = avg_chunk[local_row, content_idx_tensor]
                row_score = pool_text_vector(row_vec, pool, pool_topk)
                frame_row_scores[frame].append(row_score)

            if save_viz_mats:
                headavg_buffers[frame].append(avg_chunk[local_row].detach().cpu().to(torch.float16))

        del q_chunk, scores, probs, avg_chunk

    frame_scores = finalize_frame_scores(frame_row_scores, score_topk)
    per_frame_json = [
        {"frame": idx, "score": score, "row_count": len(frame_row_scores[idx])}
        for idx, score in enumerate(frame_scores)
    ]

    if save_viz_mats:
        for frame in range(num_frames):
            if headavg_buffers[frame]:
                mat = torch.stack(headavg_buffers[frame], dim=0)
            else:
                mat = torch.zeros((0, text_idx.numel()), dtype=torch.float16)
            npy_path = viz_dir / f"viz_frame_{frame:03d}_headavg.npy"
            np.save(npy_path, mat.numpy())
            png_path = viz_dir / f"viz_frame_{frame:03d}_headavg.png"
            save_heatmap(mat.numpy(), png_path, f"Frame {frame} V→T")

    return frame_scores, per_frame_json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute layer-0 QK attention for Qwen3-VL.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--chunk_q", type=int, default=1024)
    parser.add_argument("--pool", type=str, default="topk", choices=("mean", "max", "topk"))
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--score_topk", type=int, default=3)
    parser.add_argument("--save_viz_mats", action="store_true")
    parser.add_argument("--no_head_avg", action="store_true")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--filter_video_pad", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and processor ...")
    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    target_device = next(model.parameters()).device

    capture = PrefillStateCapture()
    capture.register(model, layer_idx=0)

    print("Preparing inputs ...")
    inputs, media_meta = prepare_inputs(
        processor=processor,
        video_path=args.video_path,
        question=args.question,
        device=target_device,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"Input ids shape: {tuple(inputs.input_ids.shape)}")

    pad_token_id = processor.tokenizer.pad_token_id or getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)

    print("Running single prefill (FlashAttention 2 remains enabled) ...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        )

    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if video_token_id is None:
        raise RuntimeError("Tokenizer missing <|video_pad|> token.")
    mapper = TokenIndexMapper(inputs.input_ids, video_token_id, inputs.video_grid_thw)
    vision_indices = mapper.vision_idx.cpu()

    text_idx = mapper.get_question_tokens(processor, filter_video_pad=args.filter_video_pad)
    if text_idx.numel() == 0:
        raise RuntimeError("Question token selection is empty after filtering.")
    text_tokens = [processor.tokenizer.decode([int(inputs.input_ids[0, idx])]).strip() for idx in text_idx]
    text_token_ids = [int(inputs.input_ids[0, idx]) for idx in text_idx]
    question_ids = processor.tokenizer(args.question, add_special_tokens=False).input_ids
    if question_ids:
        span_start = _find_subsequence(text_token_ids, question_ids)
        if span_start is not None:
            span_end = span_start + len(question_ids)
            text_idx = text_idx[span_start:span_end]
            text_tokens = text_tokens[span_start:span_end]
            text_token_ids = text_token_ids[span_start:span_end]
    content_mask_flags = build_content_mask(
        processor.tokenizer,
        text_token_ids,
        text_tokens,
        question=args.question,
    )
    content_indices = [i for i, flag in enumerate(content_mask_flags) if flag]
    if not content_indices:
        content_indices = list(range(len(text_idx)))
    content_idx_tensor = torch.tensor(content_indices, device=target_device, dtype=torch.long)

    print(
        f"Text tokens: {len(text_idx)} | content tokens: {content_idx_tensor.numel()} | frames: {mapper.num_frames}"
    )

    frame_scores, per_frame_json = stream_qk_attention(
        attn_module=_resolve_layers_container(model)[0].self_attn,
        capture=capture,
        vision_indices=vision_indices,
        text_indices=text_idx,
        mapper=mapper,
        content_idx_tensor=content_idx_tensor,
        chunk_q=args.chunk_q,
        pool=args.pool,
        pool_topk=args.topk,
        score_topk=args.score_topk,
        save_viz_mats=args.save_viz_mats,
        no_head_avg=args.no_head_avg,
        output_dir=output_dir,
    )
    capture.clear()

    frame_scores_tensor = torch.tensor(frame_scores, dtype=torch.float32)
    if frame_scores_tensor.numel():
        stats = {
            "min": float(frame_scores_tensor.min().item()),
            "median": float(frame_scores_tensor.median().item()),
            "max": float(frame_scores_tensor.max().item()),
        }
    else:
        stats = {"min": 0.0, "median": 0.0, "max": 0.0}
    print(
        "Frame score stats: min={min:.4f} median={median:.4f} max={max:.4f}".format(**stats)
    )

    np.save(output_dir / "per_frame_scores.npy", frame_scores_tensor.cpu().numpy())
    (output_dir / "per_frame_scores.json").write_text(json.dumps(per_frame_json, indent=2))

    meta = {
        "model_path": args.model_path,
        "video_path": args.video_path,
        "question": args.question,
        "chunk_q": args.chunk_q,
        "pool": args.pool,
        "pool_topk": args.topk,
        "score_topk": args.score_topk,
        "no_head_avg": bool(args.no_head_avg),
        "save_viz_mats": bool(args.save_viz_mats),
        "text_token_count": len(text_idx),
        "content_token_count": int(content_idx_tensor.numel()),
        "num_frames": mapper.num_frames,
        "stats": stats,
        "media_meta": media_meta,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
