"""Build the config dict sent to ArcticRL's grpo_loss on each /fwd-bwd call."""

from __future__ import annotations

from loguru import logger

_CISPO_LOGGED = False


def build_grpo_loss_config(
    current_version: int,
    eps_clip: float = 0.2,
    use_cispo: bool = False,
    loss_agg_mode: str = "token-mean",
) -> dict:
    """Return the `config` sub-dict to send under `processing={"config": ...}`.

    Keys map 1:1 to ArcticRL's `grpo_loss` signature in
    ArcticTraining-dss/arctic_training/arctic_rl/processors/grpo.py.
    """
    global _CISPO_LOGGED
    if use_cispo:
        if not _CISPO_LOGGED:
            msg = f"LOSS CONFIG: CISPO enabled (eps=0.2/0.28, prox=recompute, agg={loss_agg_mode})"
            logger.info(msg)
            print(msg, flush=True)
            _CISPO_LOGGED = True
        return {
            "use_cispo_loss": True,
            "eps_clip": eps_clip,
            "eps_clip_higher": 0.28,
            "loss_agg_mode": loss_agg_mode,
            # "recompute" returns old_log_probs when prox_logp_gt is None
            # (see _resolve_proximal_logp:102-106 in ArcticTraining-dss).
            "prox_logp_method": "recompute",
            "importance_sampling_level": "token",
            "current_version": current_version,
        }

    if not _CISPO_LOGGED:
        logger.info(f"LOSS CONFIG: vanilla PPO (agg={loss_agg_mode})")
        _CISPO_LOGGED = True
    return {
        "eps_clip": eps_clip,
        "loss_agg_mode": loss_agg_mode,
        "prox_logp_method": "recompute",
        "importance_sampling_level": "token",
        "current_version": current_version,
    }
