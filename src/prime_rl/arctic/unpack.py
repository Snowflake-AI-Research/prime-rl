"""Unpack a Prime-RL packed `[1, T]` microbatch to `[B, S]` padded form.

Prime-RL's DataLoader packs multiple rollouts into one sequence via
`packed_samples_into_micro_bs` (src/prime_rl/trainer/batch.py:87) and unsqueezes
to shape `[1, T]`. DSS's `get_per_replica_batches` splits tensors along dim 0
via `torch.chunk(dim=0, world_size)` (dss-platform/dss/job_manager/training.py:79),
which asserts a chunk count equal to `world_size` and therefore fails for `[1, T]`
when world_size > 1.

This module converts the packed representation back to `[B, S]` padded form with
an accompanying `attention_mask`, enabling DSS DP>1. The server already supports
the unpacked path — `run_pipeline` auto-invokes `pack_sequences` when `cu_seqlens`
is absent from context (ArcticTraining-dss/arctic_training/arctic_rl/processors/
pipeline.py:153-163).
"""

from __future__ import annotations

import torch

# Pad values match Prime-RL's own `pad_micro_batch` (trainer/batch.py:163-176)
# so the unpack is numerically indistinguishable from a natively padded batch.
# `loss_mask=False` at padded positions is the authoritative guard; every
# downstream op respects it, so the other pad values rarely matter.
_PAD_VALUES: dict[str, float | int | bool] = {
    "input_ids": 1,
    "position_ids": 0,
    "advantages": 0.0,
    "inference_logprobs": 0.0,
    "old_log_probs_shifted": 0.0,
    "teacher_logprobs": 0.0,
    "teacher_log_probs_shifted": 0.0,
    "loss_mask": False,
    "temperatures": 1.0,
}


def _detect_rollout_starts(position_ids: torch.Tensor) -> list[int]:
    """Return rollout start indices for a packed `[1, T]` position_ids tensor.

    A boundary exists wherever `position_ids[i] - position_ids[i-1] != 1`, plus
    position 0. This correctly handles length-1 rollouts (where the naive
    `pos==0 AND prev!=0` detector from `trainer/utils.py:get_response_lengths`
    would miss the transition).
    """
    assert position_ids.dim() == 2 and position_ids.shape[0] == 1, (
        f"Expected position_ids of shape [1, T], got {tuple(position_ids.shape)}"
    )
    pos_flat = position_ids.squeeze(0)
    t = pos_flat.shape[0]
    is_start = torch.zeros(t, dtype=torch.bool, device=pos_flat.device)
    is_start[0] = True
    is_start[1:] = pos_flat[1:] - pos_flat[:-1] != 1
    return is_start.nonzero(as_tuple=True)[0].tolist()


def iter_rollout_slices(mb: dict) -> list[tuple[int, int]]:
    """Return per-rollout `(start, end)` slice indices into a packed `[1, T]` mb.

    Drops the trailing pad segment (Prime-RL's `pad_micro_batch` appends one
    with loss_mask=False throughout). Guarded so a fully-masked real rollout
    isn't silently dropped from a non-trailing position.
    """
    position_ids = mb["position_ids"]
    loss_mask = mb["loss_mask"]

    t = position_ids.shape[1]
    starts = _detect_rollout_starts(position_ids)
    ends = starts[1:] + [t]
    slices = list(zip(starts, ends))

    loss_mask_flat = loss_mask.squeeze(0)
    if len(slices) > 1 and not loss_mask_flat[slices[-1][0] : slices[-1][1]].any():
        slices = slices[:-1]

    assert len(slices) >= 1, "Expected at least one rollout after dropping trailing pad"
    return slices


def unpack_packed_microbatch(rolled: dict) -> dict:
    """Convert a `[1, T]` packed microbatch dict to `[B, S]` padded form.

    Args:
        rolled: dict whose `[1, T]` tensor values are per-token fields already
            aligned as the caller intends (e.g. `torch.roll(-1)` already applied).
            Must contain `position_ids` (used for boundary detection) and
            `loss_mask` (used to drop a trailing pad segment, if present).

    Returns:
        dict with the same keys, each `[1, T]` tensor replaced by a `[B, S]`
        right-padded tensor, plus a freshly computed `attention_mask`.
        Non-tensor values and tensors not shaped `[1, T]` pass through unchanged.
    """
    position_ids = rolled["position_ids"]
    loss_mask = rolled["loss_mask"]

    t = position_ids.shape[1]
    starts = _detect_rollout_starts(position_ids)
    n_segs = len(starts)
    ends = starts[1:] + [t]
    lengths = [e - s for s, e in zip(starts, ends)]

    # Drop trailing pad segment (Prime-RL's `pad_micro_batch` appends one with
    # loss_mask=False throughout). Guarded so a fully-masked real rollout
    # wouldn't be silently dropped from a non-trailing position.
    loss_mask_flat = loss_mask.squeeze(0)
    if n_segs > 1 and not loss_mask_flat[starts[-1] : ends[-1]].any():
        starts = starts[:-1]
        ends = ends[:-1]
        lengths = lengths[:-1]

    assert len(lengths) >= 1, "Expected at least one rollout after dropping trailing pad"

    b = len(lengths)
    s = max(lengths)

    out: dict = {}
    for key, value in rolled.items():
        if not isinstance(value, torch.Tensor) or value.dim() != 2 or value.shape[0] != 1:
            out[key] = value
            continue
        pad_val = _PAD_VALUES.get(key, 0)
        flat = value.squeeze(0)
        padded = torch.full((b, s), pad_val, dtype=value.dtype, device=value.device)
        for i, (seg_start, seg_len) in enumerate(zip(starts, lengths)):
            padded[i, :seg_len] = flat[seg_start : seg_start + seg_len]
        out[key] = padded

    attention_mask = torch.zeros((b, s), dtype=position_ids.dtype, device=position_ids.device)
    for i, seg_len in enumerate(lengths):
        attention_mask[i, :seg_len] = 1
    out["attention_mask"] = attention_mask

    return out
