"""Arctic trainer adapter.

Replaces Prime-RL's native torchrun+FSDP2 trainer with a single-process loop
that issues /fwd-bwd + /step + /sync-weights over HTTP to a DSS server.
Reuses Prime-RL's DataLoader for the orchestrator->trainer transport.

- One /fwd-bwd per training step (the server splits into microbatches
  internally; the client sends one consolidated batch).
- Client-side LR scheduling: a dummy optimizer drives trainer_cfg.scheduler;
  the computed LR is passed as adam_params on each /step call, overriding the
  server's own scheduler.
- Weight sync: client.sync_weights() + write STABLE marker to preserve
  Prime-RL's orchestrator polling contract.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from prime_rl.arctic.client import build_arctic_client
from prime_rl.configs.arctic import ArcticConfig
from prime_rl.arctic.context import microbatches_to_arctic_context
from prime_rl.arctic.loss_config import build_grpo_loss_config
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.scheduler import setup_scheduler


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_single_process_dist() -> None:
    """Initialize a world_size=1 gloo process group so DataLoader can use get_world()."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _write_stable_marker(broadcast_dir: Path, step: int) -> None:
    """Mimic FileSystemWeightBroadcast._notify_orchestrator: zero-byte STABLE file.

    The orchestrator polls <output_dir>/broadcasts/step_<N>/STABLE via
    get_weight_dir() before starting rollouts for step N. Writing this marker
    is what gates the orchestrator's async_level tracking in Arctic mode; the
    actual weight sync already happened server-side via ArcticRLClient.sync_weights().
    """
    stable_path = broadcast_dir / f"step_{step}" / "STABLE"
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    stable_path.touch()


def _concat_microbatches(mbs: list[dict]) -> dict:
    """Concatenate a list of TensorMicroBatches along the batch dim.

    Prime-RL's DataLoader yields a list (one microbatch per DP-local minibatch).
    Arctic sends one consolidated batch per /fwd-bwd; the server re-splits by
    its own mb_spec token budget.
    """
    if len(mbs) == 1:
        return dict(mbs[0])

    out: dict = {}
    for key, value in mbs[0].items():
        if value is None:
            out[key] = None
            continue
        if isinstance(value, torch.Tensor):
            out[key] = torch.cat([mb[key] for mb in mbs], dim=0)
        else:
            # Non-tensor payloads (e.g., routed_experts may be None across mbs)
            out[key] = value
    return out


class ArcticTrainerAdapter:
    """Single-process trainer that delegates forward/backward to a DSS server.

    Wires Prime-RL's DataLoader (unchanged) to ArcticRLClient's /fwd-bwd,
    /step, and /sync-weights endpoints. Writes Prime-RL's STABLE marker after
    each sync so the orchestrator's polling loop proceeds unchanged.
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

        # <output_dir>/run_default/broadcasts matches native Prime-RL path
        # (see src/prime_rl/trainer/rl/broadcast/filesystem.py:107 and
        # src/prime_rl/orchestrator/utils.py:114).
        self.broadcast_dir = Path(trainer_cfg.output_dir) / "run_default" / "broadcasts"

        # Initialize judge sampling job AFTER the main client.
        # ArcticRLClient._cleanup_stale_jobs() destroys all running jobs at
        # the start of __init__; initializing the judge before that would
        # cause it to be destroyed immediately.
        # judge_url defaults to the rollout zone URL. Setting it to a second
        # zone (e.g., on another node) sidesteps DSS's single-node Ray model
        # (sampling.py:_init_driver) without any multi-node NCCL.
        judge_sampling_job_id: int | None = None
        judge_dss_url: str | None = None
        if arctic_cfg.judge_model_name:
            import time as _time

            import requests as _req

            judge_dss_url = arctic_cfg.judge_url or arctic_cfg.url or "http://localhost:7000"

            # When judge_url points at a second zone (different host from the
            # main ArcticRLClient's url), _cleanup_stale_jobs() won't touch
            # its jobs. Stale judges from prior runs accumulate and the
            # ArcticInference Driver's _compute_even_share splits GPUs across
            # ALL registered models, so the new judge gets only 1 replica
            # when there are 7 dead ones from previous runs. Clean up
            # explicitly here before creating the new judge.
            if judge_dss_url != (arctic_cfg.url or "http://localhost:7000"):
                try:
                    status = _req.get(f"{judge_dss_url}/status", timeout=10).json()
                    for stale_id, info in status.get("jobs", {}).items():
                        if info.get("job_type") == "sampling":
                            logger.info(
                                "Destroying stale judge-zone job {} on {}",
                                stale_id,
                                judge_dss_url,
                            )
                            _req.post(
                                f"{judge_dss_url}/destroy",
                                params={"job_id": stale_id},
                                json={"job_type": "sampling"},
                                timeout=60,
                            )
                except Exception as exc:
                    logger.warning("Judge-zone cleanup skipped: {}", exc)

            vllm_config = arctic_cfg.judge_vllm_config or {"dtype": "bfloat16", "trust_remote_code": True}
            # No HTTP timeout: DSS driver.initialize() loads the model synchronously
            # and can take 10-20 min for a cold 8B download. Let the poll loop below
            # handle readiness; raising immediately only on FAILED/CANCELED.
            resp = _req.post(
                f"{judge_dss_url}/initialize",
                json={"model_name": arctic_cfg.judge_model_name, "job_type": "sampling", "vllm_config": vllm_config},
            )
            resp.raise_for_status()
            judge_sampling_job_id = resp.json()["job_id"]
            logger.info(
                "Judge sampling job_id={} queued on {}, waiting for RUNNING…",
                judge_sampling_job_id,
                judge_dss_url,
            )
            # No fixed timeout: raise immediately on FAILED/CANCELED, keep waiting otherwise.
            # DSS job_timeout_seconds (configured on the server) acts as the runaway guard.
            while True:
                r = _req.get(f"{judge_dss_url}/job/{judge_sampling_job_id}", timeout=5)
                if r.ok:
                    status = r.json().get("status", "")
                    if status == "RUNNING":
                        logger.info("Judge job {} is RUNNING", judge_sampling_job_id)
                        break
                    if status in ("FAILED", "CANCELED"):
                        raise RuntimeError(f"Judge job {judge_sampling_job_id} reached status {status}")
                _time.sleep(5)

        # Write reconnect config so the launcher can hand it to the shim(s)
        # without stdout parsing. ArcticRLClientConfig marks job ID fields as
        # Field(exclude=True), so model_dump_json() silently drops them —
        # we build the payload manually to preserve all IDs.
        reconnect_path = Path(trainer_cfg.output_dir) / "configs" / "reconnect.json"
        reconnect_path.parent.mkdir(parents=True, exist_ok=True)
        rc = self.client.reconnect_config()
        import json as _json
        from urllib.parse import urlparse as _urlparse

        judge_host: str | None = None
        judge_port: int | None = None
        if judge_dss_url:
            parsed = _urlparse(judge_dss_url)
            judge_host = parsed.hostname
            judge_port = parsed.port

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
                    "judge_sampling_job_id": judge_sampling_job_id,
                    "judge_model_name": arctic_cfg.judge_model_name,
                    "judge_host": judge_host,
                    "judge_port": judge_port,
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

            # Consolidate all microbatches' rollouts into ONE `[B_total, max_S]`
            # padded batch and issue a single /fwd-bwd call. This keeps B_total
            # equal to the step's total number of real rollouts (typically
            # batch_size after zero-advantage filtering — tens to hundreds),
            # so DSS's `torch.chunk(dim=0, world_size)` always produces
            # `world_size` chunks. The server's `pack_sequences` repacks each
            # DP rank's shard into a variable-length `[1, T]` before the model
            # forward, so activation memory is the same as a per-microbatch
            # packed call.
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
            step_result = self.client.step(learning_rate=current_lr) or {}

            if step > 0:
                logger.info("Step {} — /sync-weights", step)
                self.client.sync_weights()
                _write_stable_marker(self.broadcast_dir, step)

            # Clear ready_to_update so SinglePacker.pack() can receive the next
            # step's batch. In native Prime-RL this is done by
            # FileSystemWeightBroadcast.broadcast_weights() (which we skip since
            # DSS owns the weights); rl/train.py:253-257 handles the same case
            # when broadcast is absent.
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
