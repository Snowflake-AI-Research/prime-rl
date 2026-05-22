"""Arctic adapter config.

`ArcticConfig` is attached to `RLConfig.arctic`. When `backend` is None, the
entire adapter is inactive and Prime-RL runs natively.
"""

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig


class ArcticConfig(BaseConfig):
    """DSS Arctic integration config.

    When `backend` is set (not None), the launcher dispatches to
    `rl_arctic_local` which replaces the native trainer and inference
    subprocesses with the Arctic adapter (HTTP to DSS) and an OpenAI-compat
    shim.
    """

    backend: Annotated[
        Literal["dss", "local"] | None,
        Field(
            description=(
                "Arctic backend: 'dss' connects to a remote DSS server at `url`; "
                "'local' spawns an `arctic_training.arctic_rl.server` subprocess "
                "via ArcticRLClient's local backend. None (default) disables "
                "Arctic and uses the native Prime-RL path."
            )
        ),
    ] = None

    url: Annotated[
        str | None,
        Field(description="Base URL (http://host:port) of the DSS server. Required when backend='dss'."),
    ] = None

    training_gpus: Annotated[
        int,
        Field(description="GPUs requested for the training job.", gt=0),
    ] = 2
    sampling_tensor_parallel_size: Annotated[
        int,
        Field(
            description=(
                "Tensor-parallel size for each sampling replica. For DSS/Ray, the sampling zone GPU count "
                "controls the number of replicas; this field is only forwarded as vLLM tensor_parallel_size."
            ),
            gt=0,
        ),
    ] = 1
    log_prob_gpus: Annotated[
        int,
        Field(description="Local backend only: GPUs for the log-prob job. 0 disables it."),
    ] = 0

    vllm_config: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional vLLM overrides forwarded to DSS sampling/log-prob engines. "
                "max_model_len defaults to trainer.model.seq_len when omitted."
            )
        ),
    ] = None

    shim_host: Annotated[str, Field(description="OpenAI-compat shim bind host.")] = "127.0.0.1"

    enable_thinking: Annotated[
        bool,
        Field(description="Pass enable_thinking to tokenizer.apply_chat_template in the OpenAI-compat shim."),
    ] = False

    judge_model_name: Annotated[
        str | None,
        Field(
            description=(
                "If set, initialize a second DSS sampling job for an LLM judge. "
                "The judge shim listens on judge_shim_port. "
                "Must be initialized after the main ArcticRLClient to avoid "
                "_cleanup_stale_jobs() destroying it."
            )
        ),
    ] = None
    judge_url: Annotated[
        str | None,
        Field(
            description=(
                "Optional DSS zone URL for the judge sampling job. Defaults to "
                "`url` (co-host judge in the same zone as rollout). Set this to "
                "target a second DSS zone on another node, which avoids Ray "
                "single-node resource contention and cross-node NCCL."
            )
        ),
    ] = None
    judge_shim_port: Annotated[
        int,
        Field(description="Port for the judge OpenAI-compat shim. Used when judge_model_name is set."),
    ] = 8011
    judge_vllm_config: Annotated[
        dict[str, Any] | None,
        Field(description="Optional vLLM overrides forwarded to the judge sampling engine."),
    ] = None

    use_cispo_loss: Annotated[
        bool,
        Field(
            description=(
                "Use CISPO (Clipped IS-weight Policy Optimization) instead of vanilla "
                "PPO-CLIP. CISPO clips the importance-sampling ratio with a stop-gradient, "
                "so every token including clipped ones contributes a non-zero gradient. "
                "Recommended for async-pipeline off-policy training (ScaleRL §3.2). "
                "Uses asymmetric clips eps=0.2 / eps_higher=0.28 per the paper."
            )
        ),
    ] = False

    loss_agg_mode: Annotated[
        str,
        Field(
            description=(
                "Loss aggregation mode forwarded to ArcticRL's grpo_loss. "
                "'token-mean' (default): average over all unmasked tokens. "
                "'prompt-mean': average per rollout first, then across rollouts — "
                "requires prompt_group_ids (automatically built when example_ids are tracked). "
                "Other valid values: 'seq-mean-token-sum', 'seq-mean-token-sum-norm', 'seq-mean-token-mean'."
            )
        ),
    ] = "token-mean"

    @model_validator(mode="after")
    def _validate(self) -> "ArcticConfig":
        if self.backend == "dss" and not self.url:
            raise ValueError("arctic.backend='dss' requires arctic.url to be set.")
        return self
