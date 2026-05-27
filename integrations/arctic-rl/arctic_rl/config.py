"""Arctic adapter config.

`ArcticConfig` is attached to `RLConfig.arctic`. When `backend` is None, the
entire adapter is inactive and Prime-RL runs natively.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from prime_rl.utils.config import BaseConfig


class ArcticConfig(BaseConfig):
    """Arctic RL integration config.

    When `backend = "local"`, the launcher dispatches to the Arctic
    entrypoint, which spawns the Arctic RL server in-process via
    `ArcticRLClient` and replaces prime-rl's native trainer + vLLM with
    HTTP calls to it.
    """

    backend: Annotated[
        Literal["local"] | None,
        Field(
            description=(
                "Arctic backend mode. 'local' spawns the Arctic RL server in-process. "
                "None (default) disables Arctic and uses the native Prime-RL path."
            )
        ),
    ] = None

    training_gpus: Annotated[
        int,
        Field(description="GPUs requested for the training job.", gt=0),
    ] = 2
    sampling_tensor_parallel_size: Annotated[
        int,
        Field(description="Tensor-parallel size for the sampling engine (forwarded as vLLM tensor_parallel_size).", gt=0),
    ] = 1
    log_prob_gpus: Annotated[
        int,
        Field(description="GPUs for the log-prob job. 0 disables it."),
    ] = 0

    vllm_config: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional vLLM overrides forwarded to Arctic RL sampling/log-prob engines. "
                "max_model_len defaults to trainer.model.seq_len when omitted."
            )
        ),
    ] = None

    enable_thinking: Annotated[
        bool,
        Field(description="Pass enable_thinking to tokenizer.apply_chat_template."),
    ] = False

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
                "Loss aggregation mode forwarded to the server's grpo_loss. "
                "'token-mean' (default): average over all unmasked tokens. "
                "'prompt-mean': average per rollout first, then across rollouts — "
                "requires prompt_group_ids (automatically built when example_ids are tracked). "
                "Other valid values: 'seq-mean-token-sum', 'seq-mean-token-sum-norm', 'seq-mean-token-mean'."
            )
        ),
    ] = "token-mean"
