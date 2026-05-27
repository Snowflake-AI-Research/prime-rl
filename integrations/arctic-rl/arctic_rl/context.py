"""Translate PRIME-RL packed microbatches to Arctic RL's batch format.

PRIME-RL packs multiple rollouts into ``[1, T]`` microbatches; Arctic RL
expects ``[B, max_S]`` with one rollout per row. We unpack, apply a
per-rollout left-shift (so labels match Arctic's shifted-index convention)
and zero the loss mask at the wrap-around position.
"""

from __future__ import annotations

import torch

from arctic_rl.unpack import iter_rollout_slices, unpack_packed_microbatch
from prime_rl.trainer.rl.data import TensorMicroBatch

# Pad values mirror PRIME-RL's pad_micro_batch so the padded batch is
# numerically indistinguishable from a natively-padded one.
_PAD_VALUES: dict[str, float | int | bool] = {
    "input_ids": 1,
    "position_ids": 0,
    "old_log_probs_shifted": 0.0,
    "advantages": 0.0,
    "teacher_log_probs_shifted": 0.0,
    "loss_mask": False,
}


def _extract_and_roll_rollout(mb: TensorMicroBatch, start: int, end: int) -> dict:
    """Extract one rollout's slice from a packed mb, applying per-rollout `torch.roll(-1)`.

    Returns 1-D tensors (no batch dim). The roll is scoped to this rollout so
    cross-rollout contamination at boundary positions is impossible.
    """
    out: dict = {
        "input_ids": mb["input_ids"][0, start:end].clone(),
        "position_ids": mb["position_ids"][0, start:end].clone(),
        "old_log_probs_shifted": torch.roll(mb["inference_logprobs"][0, start:end], shifts=-1, dims=-1),
        "advantages": torch.roll(mb["advantages"][0, start:end], shifts=-1, dims=-1),
    }
    loss_mask = torch.roll(mb["loss_mask"][0, start:end], shifts=-1, dims=-1).clone()
    # The last position wraps around within this rollout; not a valid target.
    loss_mask[-1] = False
    out["loss_mask"] = loss_mask

    teacher = mb.get("teacher_logprobs")
    if teacher is not None:
        out["teacher_log_probs_shifted"] = torch.roll(teacher[0, start:end], shifts=-1, dims=-1)
    return out


def microbatches_to_arctic_context(mbs: list[TensorMicroBatch]) -> dict:
    """Consolidate all rollouts from a list of packed microbatches into one `[B, max_S]` batch.

    Output:
        A dict of `[B_total, max_S]` tensors where `B_total = sum of rollouts
        across all mbs` (trailing pad per mb is dropped) and `max_S` is the
        longest rollout across the batch. Includes a real `attention_mask`.
    """
    assert mbs, "Expected at least one microbatch"

    rollouts: list[dict] = []
    raw_example_ids: list[int] = []
    for mb in mbs:
        slices = iter_rollout_slices(mb)
        mb_ids = mb.get("example_ids") or []
        for i, (start, end) in enumerate(slices):
            rollouts.append(_extract_and_roll_rollout(mb, start, end))
            if i < len(mb_ids):
                raw_example_ids.append(mb_ids[i])

    b = len(rollouts)
    max_s = max(r["input_ids"].shape[0] for r in rollouts)
    ref = rollouts[0]

    out: dict = {}
    for key, ref_tensor in ref.items():
        pad_val = _PAD_VALUES.get(key, 0)
        tensor = torch.full((b, max_s), pad_val, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for i, r in enumerate(rollouts):
            length = r[key].shape[0]
            tensor[i, :length] = r[key]
        out[key] = tensor

    attention_mask = torch.zeros((b, max_s), dtype=ref["input_ids"].dtype, device=ref["input_ids"].device)
    for i, r in enumerate(rollouts):
        attention_mask[i, : r["input_ids"].shape[0]] = 1
    out["attention_mask"] = attention_mask

    # Build prompt_group_ids for prompt-mean loss aggregation. Maps each rollout row to its
    # example group so agg_loss("prompt-mean") averages token losses per rollout before
    # averaging across rollouts. Only present when all rollouts carried an example_id.
    if len(raw_example_ids) == b:
        out["prompt_group_ids"] = torch.tensor(raw_example_ids, dtype=torch.long)

    return out


def microbatch_to_arctic_context(mb: TensorMicroBatch) -> dict:
    """Legacy single-microbatch path kept for unit tests.

    Prefer `microbatches_to_arctic_context([mb, ...])` in the training loop
    so that a single-rollout bin doesn't trip the server's DP-chunk assertion.
    """
    loss_mask = torch.roll(mb["loss_mask"], shifts=-1, dims=-1).clone()
    loss_mask[..., -1] = False

    rolled: dict = {
        "input_ids": mb["input_ids"],
        "position_ids": mb["position_ids"],
        "old_log_probs_shifted": torch.roll(mb["inference_logprobs"], shifts=-1, dims=-1),
        "advantages": torch.roll(mb["advantages"], shifts=-1, dims=-1),
        "loss_mask": loss_mask,
    }
    teacher = mb.get("teacher_logprobs")
    if teacher is not None:
        rolled["teacher_log_probs_shifted"] = torch.roll(teacher, shifts=-1, dims=-1)

    return unpack_packed_microbatch(rolled)
