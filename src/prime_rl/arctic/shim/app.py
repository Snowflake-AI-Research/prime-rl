"""FastAPI application for the OpenAI-compat shim."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from loguru import logger


def _render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    enable_thinking: bool,
) -> str:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        template_kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def create_app(
    reconnect_config_path: Path,
    model_name: str | None = None,
    enable_thinking: bool = False,
) -> FastAPI:
    """Construct the shim app, reconnecting to an existing DSS sampling job.

    Args:
        reconnect_config_path: Path to reconnect.json written by arctic-trainer.
            Contains host, port, backend, model_name, and all three job IDs.
            The shim calls ArcticRLClient(reconnect_cfg), which skips /initialize
            and reuses the sampling job the trainer already created.
        model_name: Optional display name for /v1/models. Defaults to the
            model_name from the reconnect config.
    """
    # Read model name eagerly so /v1/models responds correctly before the
    # async _blocking_init completes. The orchestrator calls /v1/models right
    # after shim /health — before the tokenizer and ArcticRLClient are loaded.
    import json

    from prime_rl.arctic.shim.translation import (
        dss_results_to_oai_response,
        extract_routing_metadata,
        oai_sampling_params,
    )

    surface_model = model_name or json.loads(reconnect_config_path.read_text()).get("model_name", "")

    # Mutable container so lifespan can populate shared state after startup.
    state: dict[str, Any] = {"tokenizer": None, "client": None, "ready": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        def _blocking_init() -> None:
            from arctic_training.arctic_rl.client import ArcticRLClient
            from arctic_training.arctic_rl.config import ArcticRLClientConfig
            from transformers import AutoTokenizer

            # ArcticRLClientConfig marks job ID fields as Field(exclude=True),
            # so model_validate_json() would leave them None and the client would
            # fall into the else-branch and call /initialize. Construct directly.
            data = json.loads(reconnect_config_path.read_text())
            reconnect_cfg = ArcticRLClientConfig(
                host=data["host"],
                port=data["port"],
                backend=data["backend"],
                model_name=data["model_name"],
                training_job_id=data.get("training_job_id"),
                sampling_job_id=data.get("sampling_job_id"),
                log_prob_job_id=data.get("log_prob_job_id"),
            )
            tokenizer_name = reconnect_cfg.model_name
            logger.info("Loading tokenizer {}…", tokenizer_name)
            state["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name)
            # ArcticRLClient detects training_job_id is set and skips /initialize.
            state["client"] = ArcticRLClient(reconnect_cfg)
            state["ready"] = True
            logger.info(
                "arctic-shim ready (sampling_job_id={}, training_job_id={})",
                reconnect_cfg.sampling_job_id,
                reconnect_cfg.training_job_id,
            )

        async def _init():
            loop = asyncio.get_event_loop()
            # All blocking work (imports + tokenizer + client) runs in a thread
            # so the event loop is never blocked and /health responds immediately.
            await loop.run_in_executor(None, _blocking_init)

        # Fire-and-forget: uvicorn binds and /health responds immediately.
        # Tokenizer + client init complete in the background before the
        # orchestrator's first rollout request arrives.
        asyncio.create_task(_init())
        yield

    app = FastAPI(title="arctic-shim", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "ready": state["ready"]}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": surface_model, "object": "model", "created": 0, "owned_by": "arctic-shim"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> dict[str, Any]:
        import asyncio

        body = await req.json()
        messages = body.get("messages", [])
        tools = body.get("tools")
        prompt = _render_chat_prompt(
            state["tokenizer"],
            messages,
            tools,
            enable_thinking=enable_thinking,
        )
        sampling = oai_sampling_params(body)
        routing_key, strict = extract_routing_metadata(body)
        logger.debug(
            "POST /v1/chat/completions (n={}, max_tokens={}, routing_key={}, strict={})",
            sampling.get("n", 1),
            sampling.get("max_tokens"),
            routing_key,
            strict,
        )
        # generate() is synchronous; run in executor so concurrent rollout
        # requests don't queue behind each other in the event loop.
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: state["client"].generate(
                prompts=[prompt],
                sampling_params=sampling,
                routing_key=routing_key,
                strict=strict,
            ),
        )
        return dss_results_to_oai_response(results, model=surface_model)

    @app.post("/v1/completions")
    async def completions(req: Request) -> dict[str, Any]:
        import asyncio

        body = await req.json()
        prompt = body.get("prompt")
        prompts = prompt if isinstance(prompt, list) else [prompt]
        sampling = oai_sampling_params(body)
        routing_key, strict = extract_routing_metadata(body)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: state["client"].generate(
                prompts=prompts,
                sampling_params=sampling,
                routing_key=routing_key,
                strict=strict,
            ),
        )
        return dss_results_to_oai_response(results, model=surface_model)

    # Prime-RL's orchestrator post-broadcast hooks. Real weight sync already
    # happened server-side via trainer's ArcticRLClient.sync_weights(); these
    # are acknowledgment-only so the orchestrator's existing polling proceeds.
    @app.post("/pause")
    async def pause() -> dict[str, Any]:
        return {"status": "ok"}

    @app.post("/resume")
    async def resume() -> dict[str, Any]:
        return {"status": "ok"}

    @app.post("/update_weights")
    async def update_weights(req: Request) -> dict[str, Any]:
        logger.debug("orchestrator /update_weights call - no-op in Arctic mode")
        return {"status": "ok"}

    @app.post("/init_broadcaster")
    async def init_broadcaster(req: Request) -> dict[str, Any]:
        return {"status": "ok"}

    return app
