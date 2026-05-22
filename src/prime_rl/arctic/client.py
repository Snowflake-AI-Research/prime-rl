"""Build and own the ArcticRLClient used by the Arctic trainer + shim.

Translates Prime-RL's ArcticConfig + TrainerConfig into an ArcticRLClientConfig
that ArcticRLClient can consume. Mirrors AReaL-dss/areal/engine/arctic/engine.py
lines 41-82 (the `ArcticTrainEngine.__init__` pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from prime_rl.configs.arctic import ArcticConfig

if TYPE_CHECKING:
    from prime_rl.configs.trainer import TrainerConfig


def _parse_url(url: str) -> tuple[str, int]:
    """Parse http://host:port into (host, port). Mirrors AReaL's _parse_server_url."""
    stripped = url.removeprefix("http://").removeprefix("https://")
    if ":" in stripped:
        host, port_str = stripped.rsplit(":", 1)
        # Port may include a path suffix; strip it off.
        port_str = port_str.split("/", 1)[0]
        return host, int(port_str)
    return stripped, 7000


def _build_training_config(trainer_cfg: TrainerConfig, arctic_cfg: ArcticConfig) -> dict:
    """Extract ArcticRL's expected training_config dict from TrainerConfig.

    ArcticRL's server initializes its optimizer + DeepSpeed engine from this
    dict at /initialize time. See ArcticTraining-dss/arctic_training/arctic_rl/
    config.py for the accepted schema.
    """
    optim = trainer_cfg.optim
    training_config: dict = {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "max_seq_len": trainer_cfg.model.seq_len,
        "n_gpus": arctic_cfg.training_gpus,
        "optimizer": {
            "lr": getattr(optim, "lr", 1e-5),
            "weight_decay": getattr(optim, "weight_decay", 0.0),
            "beta1": getattr(optim, "betas", (0.9, 0.999))[0] if hasattr(optim, "betas") else 0.9,
            "beta2": getattr(optim, "betas", (0.9, 0.999))[1] if hasattr(optim, "betas") else 0.999,
            "eps": getattr(optim, "eps", 1e-8),
            "gradient_clipping": 1.0,
            "lr_scheduler_type": "constant",
            "warmup_steps_proportion": 0.0,
        },
    }
    return training_config


def _build_vllm_config(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig) -> dict:
    vllm_config = dict(arctic_cfg.vllm_config or {})
    vllm_config.setdefault("max_model_len", trainer_cfg.model.seq_len)
    vllm_config.setdefault("tensor_parallel_size", arctic_cfg.sampling_tensor_parallel_size)
    return vllm_config


def build_arctic_client(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig):
    """Construct an ArcticRLClient for the given Arctic + Trainer configs.

    Blocks until DSS training + sampling (+ optional log-prob) jobs are RUNNING.
    """
    # Import lazily so `from prime_rl.configs.arctic import ArcticConfig` in
    # configs/rl.py doesn't pull arctic_training into every Prime-RL invocation.
    from arctic_training.arctic_rl.client import ArcticRLClient
    from arctic_training.arctic_rl.config import ArcticRLClientConfig

    model_name = trainer_cfg.model.name
    training_config = _build_training_config(trainer_cfg, arctic_cfg)
    vllm_config = _build_vllm_config(arctic_cfg, trainer_cfg)

    if arctic_cfg.backend == "dss":
        if not arctic_cfg.url:
            raise ValueError("arctic.backend='dss' requires arctic.url")
        host, port = _parse_url(arctic_cfg.url)
        client_config = ArcticRLClientConfig(
            host=host,
            port=port,
            backend="dss-platform",
            model_name=model_name,
            training_config=training_config,
            vllm_config=vllm_config,
        )
    elif arctic_cfg.backend == "local":
        client_config = ArcticRLClientConfig(
            backend="local",
            model_name=model_name,
            training_config=training_config,
            vllm_config=vllm_config,
            training_gpus=arctic_cfg.training_gpus,
            sampling_gpus=arctic_cfg.sampling_tensor_parallel_size,
            log_prob_gpus=arctic_cfg.log_prob_gpus,
        )
    else:
        raise ValueError(f"Unsupported arctic.backend: {arctic_cfg.backend!r}")

    logger.info(
        "Initializing ArcticRLClient (backend={}, model={}) — blocks until all jobs RUNNING",
        arctic_cfg.backend,
        model_name,
    )
    client = ArcticRLClient(client_config)
    logger.info(
        "ArcticRLClient ready: training_job_id={} sampling_job_id={} log_prob_job_id={}",
        client.training_job_id,
        client.sampling_job_id,
        getattr(client, "log_prob_job_id", None),
    )
    return client
