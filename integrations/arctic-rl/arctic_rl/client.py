"""Build the ArcticRLClient that the trainer uses to talk to the Arctic RL server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from arctic_rl.config import ArcticConfig

if TYPE_CHECKING:
    from prime_rl.configs.trainer import TrainerConfig


def _build_training_config(trainer_cfg: TrainerConfig, arctic_cfg: ArcticConfig) -> dict:
    """Build the dict the Arctic RL server expects at /initialize time."""
    optim = trainer_cfg.optim
    return {
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


def _build_vllm_config(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig) -> dict:
    vllm_config = dict(arctic_cfg.vllm_config or {})
    vllm_config.setdefault("max_model_len", trainer_cfg.model.seq_len)
    vllm_config.setdefault("tensor_parallel_size", arctic_cfg.sampling_tensor_parallel_size)
    return vllm_config


def build_arctic_client(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig):
    """Build and return an ArcticRLClient. Blocks until all jobs are RUNNING."""
    # Lazy-imported so the integration package doesn't pull arctic_training
    # at module-load time.
    from arctic_training.arctic_rl.client import ArcticRLClient
    from arctic_training.arctic_rl.config import ArcticRLClientConfig

    model_name = trainer_cfg.model.name
    client_config = ArcticRLClientConfig(
        backend="local",
        model_name=model_name,
        training_config=_build_training_config(trainer_cfg, arctic_cfg),
        vllm_config=_build_vllm_config(arctic_cfg, trainer_cfg),
        training_gpus=arctic_cfg.training_gpus,
        sampling_gpus=arctic_cfg.sampling_tensor_parallel_size,
        log_prob_gpus=arctic_cfg.log_prob_gpus,
    )
    logger.info("Initializing ArcticRLClient (model={})", model_name)
    client = ArcticRLClient(client_config)
    logger.info(
        "ArcticRLClient ready: training={} sampling={} log_prob={}",
        client.training_job_id,
        client.sampling_job_id,
        getattr(client, "log_prob_job_id", None),
    )
    return client
