"""Arctic trainer adapter.

Replaces PRIME-RL's torchrun+FSDP2 trainer with a single CPU process that
issues ``/fwd-bwd``, ``/step``, and ``/sync-weights`` over HTTP to the
Arctic RL server. Reuses PRIME-RL's DataLoader and writes the STABLE
marker after each weight sync so the orchestrator's polling loop is
unchanged.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from arctic_rl.client import build_arctic_client
from arctic_rl.config import ArcticConfig
from arctic_rl.context import microbatches_to_arctic_context
from arctic_rl.loss_config import build_grpo_loss_config
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.scheduler import setup_scheduler


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_single_process_dist() -> None:
    """world_size=1 gloo group so PRIME-RL's DataLoader can call get_world()."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _write_stable_marker(broadcast_dir: Path, step: int) -> None:
    """Write the zero-byte STABLE file the orchestrator polls before each step."""
    stable_path = broadcast_dir / f"step_{step}" / "STABLE"
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    stable_path.touch()


def _concat_microbatches(mbs: list[dict]) -> dict:
    """Concatenate the DataLoader's per-DP microbatches into one consolidated batch."""
    if len(mbs) == 1:
        return dict(mbs[0])

    out: dict = {}
    for key, value in mbs[0].items():
        if value is None:
            out[key] = None
        elif isinstance(value, torch.Tensor):
            out[key] = torch.cat([mb[key] for mb in mbs], dim=0)
        else:
            out[key] = value
    return out


class ArcticTrainerAdapter:
    """Single-process trainer that delegates fwd/bwd/optimizer to Arctic RL.

    Reuses PRIME-RL's DataLoader and scheduler unchanged; writes the
    STABLE marker after each ``sync_weights`` so the orchestrator's
    polling loop is unaffected.
    """

    def __init__(self, trainer_cfg: TrainerConfig, arctic_cfg: ArcticConfig):
        self.trainer_cfg = trainer_cfg
        self.arctic_cfg = arctic_cfg

        _init_single_process_dist()

        from transformers import AutoTokenizer

        from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
        from prime_rl.trainer.runs import Progress, setup_multi_run_manager

        self.progress = Progress()

        # MultiRunManager is a singleton DataLoader depends on.
        setup_multi_run_manager(
            output_dir=trainer_cfg.output_dir,
            max_runs=trainer_cfg.max_concurrent_runs,
            device=torch.device("cpu"),
        )

        if trainer_cfg.data.fake is not None:
            self.loader = FakeDataLoader(
                config=trainer_cfg.data.fake,
                seq_len=trainer_cfg.data.fake.seq_len if hasattr(trainer_cfg.data.fake, "seq_len") else 512,
                dp_world_size=1,
            )
            self._using_fake = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer.name)
            self.loader = DataLoader(
                output_dir=trainer_cfg.output_dir,
                start_step=self.progress.step,
                dp_world_size=1,
                seq_len=trainer_cfg.data.seq_len if hasattr(trainer_cfg.data, "seq_len") else 2048,
                pad_to_multiple_of=trainer_cfg.data.pad_to_multiple_of
                if hasattr(trainer_cfg.data, "pad_to_multiple_of")
                else 64,
                tokenizer=tokenizer,
                config=trainer_cfg.rollout_transport,
            )
            self._using_fake = False

        self.client = build_arctic_client(arctic_cfg, trainer_cfg)

        # Dummy single-parameter optimizer used only to drive the LR schedule.
        # The actual optimizer lives server-side; we pass the computed LR via
        # adam_params on each /step call, which overrides the server's own LR.
        _dummy = torch.nn.Parameter(torch.zeros(1))
        _optim = torch.optim.AdamW([_dummy], lr=trainer_cfg.optim.lr)
        self._lr_scheduler = setup_scheduler(
            _optim, trainer_cfg.scheduler, trainer_cfg.max_steps or 1, trainer_cfg.optim.lr
        )
        self._optim = _optim

        # <output_dir>/run_default/broadcasts matches PRIME-RL's native
        # broadcast directory layout.
        self.broadcast_dir = Path(trainer_cfg.output_dir) / "run_default" / "broadcasts"

        # ArcticRLClientConfig marks job ID fields as Field(exclude=True),
        # so model_dump_json() drops them — build the payload manually.
        reconnect_path = Path(trainer_cfg.output_dir) / "configs" / "reconnect.json"
        reconnect_path.parent.mkdir(parents=True, exist_ok=True)
        rc = self.client.reconnect_config()
        import json as _json

        reconnect_path.write_text(
            _json.dumps(
                {
                    "host": rc.host,
                    "port": rc.port,
                    "backend": rc.backend,
                    "model_name": rc.model_name,
                    "training_job_id": rc.training_job_id,
                    "sampling_job_id": rc.sampling_job_id,
                    "log_prob_job_id": rc.log_prob_job_id,
                }
            )
        )
        logger.info("Wrote reconnect config → {}", reconnect_path)

    def run(self) -> None:
        max_steps = self.trainer_cfg.max_steps or 100
        while self.progress.step < max_steps:
            step = self.progress.step
            logger.info("Step {} — waiting for batch", step)
            self.loader.wait_for_batch()
            mbs = self.loader.get_batch()

            processing = {
                "loss_fn": "arctic_training.arctic_rl.processors.grpo_loss",
                "config": build_grpo_loss_config(
                    current_version=step,
                    use_cispo=self.arctic_cfg.use_cispo_loss,
                    loss_agg_mode=self.arctic_cfg.loss_agg_mode,
                ),
                "post": ["compute_logprobs"],
            }

            # Send all rollouts in one consolidated [B_total, max_S] batch
            # so the server's torch.chunk(dim=0, world_size) splits cleanly.
            # The server repacks per DP shard before the model forward, so
            # activation memory matches a per-microbatch packed call.
            kwargs = microbatches_to_arctic_context(mbs)
            logger.info(
                "Step {} — /fwd-bwd (B={}, S={}, across {} source microbatch(es))",
                step,
                kwargs["input_ids"].shape[0],
                kwargs["input_ids"].shape[1],
                len(mbs),
            )
            # Write the scheduled LR before fwd-bwd so the orchestrator can
            # read it during rollout generation for this same step.
            self._lr_scheduler.step()
            current_lr = self._optim.param_groups[0]["lr"]
            (self.broadcast_dir.parent / "last_lr").write_text(str(current_lr))

            result = self.client.fwd_bwd({"args": (), "kwargs": kwargs}, processing=processing)
            avg_loss = result.get("avg_loss") or result.get("loss") or float("nan")
            logger.info("Step {} — avg_loss={:.4f}", step, avg_loss)

            logger.info("Step {} — /step", step)
            # Older ArcticRLClient.step() takes no kwargs; newer accepts learning_rate.
            import inspect as _inspect
            try:
                if "learning_rate" in _inspect.signature(self.client.step).parameters:
                    step_result = self.client.step(learning_rate=current_lr) or {}
                else:
                    step_result = self.client.step() or {}
            except (TypeError, ValueError):
                step_result = self.client.step() or {}

            if step > 0:
                logger.info("Step {} — /sync-weights", step)
                self.client.sync_weights()
                _write_stable_marker(self.broadcast_dir, step)

            # Clear ready_to_update for the next step. Native PRIME-RL does
            # this via FileSystemWeightBroadcast.broadcast_weights() which we
            # skip (Arctic owns the weights).
            if not self._using_fake:
                mgr = get_multi_run_manager()
                for idx in mgr.used_idxs:
                    mgr.ready_to_update[idx] = False

            self.progress.step += 1
            logger.info(
                "Step {} done — last_lr={} grad_norm={}",
                step,
                step_result.get("last_lr"),
                step_result.get("grad_norm"),
            )

        logger.success("Training finished after {} steps", self.progress.step)
