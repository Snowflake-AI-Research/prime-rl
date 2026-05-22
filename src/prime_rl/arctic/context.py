"""Remap Prime-RL's TensorMicroBatch list to ArcticRL's context dict.

Prime-RL's DataLoader yields a list of packed `[1, T]` microbatches per step
(one bin per FFD pack). This module consolidates all rollouts from all
microbatches into a single `[B_total, max_S]` padded batch suitable for one
`/fwd-bwd` call:

- **Shape**: DSS's `get_per_replica_batches` splits along dim 0 via
  `torch.chunk(dim=0, world_size)`. Sending `[128, max_S]` for a batch of
  128 rollouts always splits cleanly across any `world_size ≪ 128`; no dummy
  rows, no per-mb B=1 edge case.
- **Alignment** (`torch.roll(-1)`): applied per-rollout (not across the
  packed `[1, T]`) so rollout boundaries don't leak values from one sample
  into the next. ArcticRL's `compute_logprobs_post` rolls `input_ids` left by
  1 to build labels; its `grpo_loss` multiplies logprobs with old_log_probs,
  advantages, and loss_mask pointwise (grpo.py:183-204). All four share the
  same shifted-left index convention. Prime-RL's TensorMicroBatch is at
  current-token alignment (trainer/rl/loss.py:83-98 documents the shift).
- **Loss at wrap position**: after per-rollout roll, each rollout's last
  position holds the first token's value (wrap-around). That wrap isn't a
  valid next-token target, so we zero `loss_mask` there.

Do not remove the roll without running the alignment sanity check described
at .claude/plans/prime-rl-integration.md in the DSS workspace.
"""

from __future__ import annotations

import torch

from prime_rl.arctic.unpack import iter_rollout_slices, unpack_packed_microbatch
from prime_rl.trainer.rl.data import TensorMicroBatch

# Pad values match Prime-RL's `pad_micro_batch` (trainer/batch.py:163-176)
# so the padded batch is numerically indistinguishable from a natively padded one.
# `loss_mask=False` at padded positions is the authoritative guard; other pad
# values rarely matter downstream because every pointwise op respects the mask.
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
    so that a single-rollout bin doesn't trip DSS's DP-chunk assertion.
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
