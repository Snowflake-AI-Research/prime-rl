# Arctic-primerl end-to-end flow

This walks through how a single Prime-RL + DSS training run executes, from
`uv run rl @ examples/arctic_reverse_text/rl.toml` to the reward number
landing in W&B.

## High-level diagram

```
┌────────────────────────────┐         ┌───────────────────────────┐
│         uv run rl          │         │     DSS zone_server       │
│    (Launcher, PID N)       │         │  HTTP on :7000 (FastAPI)  │
│  entrypoints/rl.py         │         │  dss-platform/dss/        │
│  → rl_arctic_local         │         │    zone_server.py         │
└─┬────────┬─────────┬───────┘         └──┬────────────────────────┘
  │        │         │                    │
  │        │         │ spawns             │ HTTP:
  │        │         │                    │   /initialize
  │        │         │                    │   /fwd-bwd  /step
  │        │         │                    │   /sync-weights
  │        │         │                    │   /generate
  │        │         ▼                    ▼
  │        │    ┌────────────────┐   ┌────────────────────────────┐
  │        │    │ arctic-trainer │   │      dss-devices (N)       │
  │        │    │   (1 proc)     │   │                            │
  │        │    │ arctic/        │   │  ┌──────────┐ ┌─────────┐  │
  │        │    │  entrypoint.py │   │  │ training │ │sampling │  │
  │        │    │  trainer.py    │   │  │ DeepSpeed│ │  vLLM   │  │
  │        │    │  (HTTP client) │   │  │ zone     │ │ zone    │  │
  │        │    └────────────────┘   │  │ (GPU 0-1)│ │(GPU 2-7)│  │
  │        │                         │  └──────────┘ └─────────┘  │
  │        │                         └───────────┬────────────────┘
  │        │                                     ▲
  │        │                                     │ NCCL weight transfer
  │        │                                     │ (intra-node)
  │        │    OpenAI-compat HTTP               │
  │        │    ┌───────────────────────┐        │
  │        └───▶│     arctic-shim       │────────┘
  │             │  FastAPI 127.0.0.1:   │ (reuses sampling_job_id
  │             │   <free_port>/v1      │  the trainer created)
  │             │  arctic/shim/app.py   │
  │             └────────────▲──────────┘
  │                          │ POST /v1/chat/completions
  │       ┌──────────────────┴─────────────────┐
  └──────▶│            orchestrator            │
          │  (verifiers + rollout generation)  │
          │    prime_rl.orchestrator.*         │
          └────────────────────────────────────┘
          rollouts → outputs/run_default/rollouts/step_N/train_rollouts.bin
                    ── arctic-trainer reads via Prime-RL's filesystem transport ──
```

**Four processes run in parallel on the same node:**

1. **Launcher** (`uv run rl @ ...`) — `rl_arctic_local` supervises the other
   three subprocesses, waits for a reconnect file written by the trainer, and
   rewires the orchestrator's `base_url` to the shim.
2. **arctic-trainer** — single-process loop that issues HTTP calls to DSS
   for training (`/initialize`, `/fwd-bwd`, `/step`, `/sync-weights`).
3. **arctic-shim** — FastAPI OpenAI-compat server that translates
   `/v1/chat/completions` into DSS `/generate` calls, reusing the sampling
   job the trainer initialized.
4. **orchestrator** — stock Prime-RL orchestrator, unaware of Arctic. It
   hits the shim URL thinking it's a vLLM endpoint; writes rollouts to the
   filesystem for the trainer.

Outside the four Python processes, a separate **DSS deployment** (on the
same node for our local setup) runs:
- `dss-zone` on port 7000 — the FastAPI router that dispatches jobs.
- N `dss-devices` on ports 8000+ — per-GPU managers that host training and
  sampling engines.

## Launcher startup

Entry point: `uv run rl ...` runs the `rl` console script declared in
`pyproject.toml`, which calls `src/prime_rl/entrypoints/rl.py:rl_local`.

### Arctic dispatch

`rl_local` checks the config and delegates to `rl_arctic_local` when
Arctic mode is enabled:

```python
# src/prime_rl/entrypoints/rl.py:101 rl_local
def rl_local(config: RLConfig):
    if config.arctic is not None and config.arctic.backend is not None:
        return rl_arctic_local(config)
    # ... legacy native path unchanged below ...
```

### What `rl_arctic_local` does

`rl_arctic_local` (line 407) does six things in order:

1. **Pick a free shim port** and set the orchestrator's `base_url` to it:
   ```python
   # src/prime_rl/entrypoints/rl.py:435-436
   shim_base_url = f"http://{arctic_cfg.shim_host}:{shim_port}/v1"
   config.orchestrator.client.base_url = [shim_base_url]
   ```
   The orchestrator never learns it's talking to a shim — it sees a
   regular OAI-compat server.

2. **Write subconfigs** for the three subprocesses (trainer.toml,
   orchestrator.toml, arctic.toml) to `outputs/configs/`.

3. **Spawn arctic-trainer** (line 486) as a subprocess with stdout+stderr
   redirected to the log file directly. No stdout pipe — readiness is
   signalled via a file, not a log line.

4. **Poll for `outputs/configs/reconnect.json`** (line ~496). The trainer
   writes this file once DSS jobs are RUNNING. It contains all job IDs plus
   host/port/model_name — a complete `ArcticRLClientConfig` snapshot. The
   launcher polls until the file appears or the trainer exits:
   ```python
   # src/prime_rl/entrypoints/rl.py
   reconnect_path = config_dir / "reconnect.json"
   while not reconnect_path.exists():
       if trainer_process.poll() is not None:
           raise RuntimeError("trainer exited before writing reconnect.json")
       time.sleep(1)
   logger.success("arctic-trainer ready (reconnect.json written)")
   ```

5. **Spawn arctic-shim** (line 564) with `ARCTIC_SHIM=1` and
   `--reconnect-config <path>`. The `ARCTIC_SHIM=1` env var tells
   `src/prime_rl/__init__.py` to skip `_compat`, cutting shim boot from
   ~30s to ~4s. The launcher waits until `/health` returns
   `{"ready": True}` (not just `200`) before starting the orchestrator,
   preventing a race where `state["client"]` is still None.

6. **Spawn orchestrator** (line 597). With `base_url` pointing at the
   shim, nothing else in the orchestrator needs to change.

The launcher then enters a monitor loop: watches all three subprocesses,
propagates errors, handles SIGTERM. When any subprocess exits non-zero,
it terminates the others.

## arctic-trainer (subprocess 1)

Entry: `arctic-trainer` console script →
`src/prime_rl/arctic/entrypoint.py:43 main`.

```python
# src/prime_rl/arctic/entrypoint.py:43-48
def main():
    set_proc_title("ArcticTrainer")
    trainer_cfg: TrainerConfig = cli(TrainerConfig)         # parse trainer.toml
    arctic_cfg = _load_arctic_config()                      # parse arctic.toml
    logger.info("ArcticTrainer starting (backend={})", arctic_cfg.backend)
    ArcticTrainerAdapter(trainer_cfg, arctic_cfg).run()
```

### ArcticTrainerAdapter.__init__

```python
# src/prime_rl/arctic/trainer.py:96-147
def __init__(self, trainer_cfg, arctic_cfg):
    self.trainer_cfg = trainer_cfg
    self.arctic_cfg = arctic_cfg

    _init_single_process_dist()                            # gloo, world_size=1

    # MultiRunManager is a Prime-RL singleton; DataLoader depends on it.
    setup_multi_run_manager(output_dir=..., max_runs=1, device="cpu")

    # Prime-RL's real DataLoader — produces packed [1, T] microbatches
    # from rollouts written by the orchestrator.
    self.loader = DataLoader(
        output_dir=trainer_cfg.output_dir,
        dp_world_size=1, seq_len=2048, tokenizer=tokenizer, ...
    )

    # This is the HTTP call that creates the DSS training + sampling jobs.
    # Blocks until both are RUNNING. Takes ~60s (vLLM model load on 6 GPUs).
    self.client = build_arctic_client(arctic_cfg, trainer_cfg)

    self.broadcast_dir = Path(output_dir) / "run_default" / "broadcasts"

    # Write reconnect.json so the launcher can start the shim without
    # parsing stdout. ArcticRLClientConfig marks job ID fields as
    # Field(exclude=True) — model_dump_json() drops them — so we build
    # the payload manually.
    rc = self.client.reconnect_config()
    reconnect_path.write_text(json.dumps({
        "host": rc.host, "port": rc.port, "backend": rc.backend,
        "model_name": rc.model_name,
        "training_job_id": rc.training_job_id,
        "sampling_job_id": rc.sampling_job_id,
        "log_prob_job_id": rc.log_prob_job_id,
    }))
    logger.info("Wrote reconnect config → {}", reconnect_path)
```

`build_arctic_client` is the HTTP job-creation path:

```python
# src/prime_rl/arctic/client.py:56 build_arctic_client
def build_arctic_client(arctic_cfg, trainer_cfg):
    from arctic_training.arctic_rl.client import ArcticRLClient
    from arctic_training.arctic_rl.config import ArcticRLClientConfig

    client_config = ArcticRLClientConfig(
        host=..., port=..., backend="dss-platform",
        model_name=trainer_cfg.model.name,
        training_config=training_config,  # lr, wd, betas, etc.
    )
    return ArcticRLClient(client_config)   # triggers /initialize sampling & training
```

`ArcticRLClient.__init__` calls `_initialize_jobs()` (at
`ArcticTraining-dss/arctic_training/arctic_rl/client.py:180`) which POSTs
`/initialize` to DSS twice — once for sampling, once for training — and
waits for both jobs to reach RUNNING state.

### ArcticTrainerAdapter.run — the training loop

```python
# src/prime_rl/arctic/trainer.py:149 run
def run(self):
    max_steps = self.trainer_cfg.max_steps or 100
    while self.progress.step < max_steps:
        step = self.progress.step

        self.loader.wait_for_batch()           # block until step_N rollouts arrive
        mbs = self.loader.get_batch()          # list of [1, T] packed microbatches

        processing = {
            "loss_fn": "arctic_training.arctic_rl.processors.grpo_loss",
            "config": build_grpo_loss_config(current_version=step),
            "post":   ["compute_logprobs"],
        }

        # Consolidate ALL step rollouts into [B_total, max_S] for one call.
        # See src/prime_rl/arctic/context.py for the unpack+roll+pad logic.
        kwargs = microbatches_to_arctic_context(mbs)
        result = self.client.fwd_bwd(
            {"args": (), "kwargs": kwargs},
            processing=processing,
        )                                       # POST /fwd-bwd (binary torch.save)

        step_result = self.client.step()        # POST /step (JSON)

        if step > 0:
            self.client.sync_weights()          # POST /sync-weights → NCCL transfer
            _write_stable_marker(self.broadcast_dir, step)   # tell orchestrator OK

        # Unblock Prime-RL's packer for the next step.
        for idx in get_multi_run_manager().used_idxs:
            get_multi_run_manager().ready_to_update[idx] = False

        self.progress.step += 1
```

Key details:
- **Packer/DataLoader interplay:** `wait_for_batch()` calls
  `packer.pack()` (Prime-RL) which reads `train_rollouts.bin` from
  `outputs/run_default/rollouts/step_N/` and writes per-rank
  `rank_K.bin` to `outputs/rollouts/step_N/`. The DataLoader then reads
  `rank_0.bin`.
- **Consolidation:** `microbatches_to_arctic_context(mbs)` unpacks every
  rollout from every packed microbatch into a single `[B_total, max_S]`
  tensor, per-rollout `torch.roll(-1)` for ArcticRL's shifted-label
  convention, pads to the longest rollout across the step.
- **step 0 skips `/sync-weights`:** the initial weights are already in the
  sampling engine from `/initialize`, so there's nothing to sync.
- **STABLE marker:** the orchestrator polls
  `outputs/run_default/broadcasts/step_N/STABLE` to know it can advance
  past its async-level gate. We touch the file after `sync_weights` since
  from the orchestrator's POV that's when new weights became "stable".
- **ready_to_update reset:** Prime-RL's `SinglePacker.pack` sets
  `ready_to_update[0] = True` after packing. Its receiver skips runs
  marked ready. Native Prime-RL clears the flag inside
  `FileSystemWeightBroadcast.broadcast_weights`, which we skip (DSS owns
  the weights). Without this manual reset, `pack()` would loop forever
  on step 1.

## arctic-shim (subprocess 2)

Entry: `arctic-shim` console script → `arctic/shim/entrypoint.py:main`
which calls `create_app` and runs uvicorn on the port the launcher chose.

### Lifespan: lazy client init via reconnect.json

```python
# src/prime_rl/arctic/shim/app.py:12 create_app
def create_app(reconnect_config_path: Path, model_name: str | None = None):
    # Read model name eagerly — /v1/models must answer before _blocking_init.
    surface_model = model_name or json.loads(reconnect_config_path.read_text())["model_name"]
    state = {"tokenizer": None, "client": None, "ready": False}

    @asynccontextmanager
    async def lifespan(app):
        def _blocking_init():
            data = json.loads(reconnect_config_path.read_text())
            reconnect_cfg = ArcticRLClientConfig(
                host=data["host"], port=data["port"],
                backend=data["backend"], model_name=data["model_name"],
                training_job_id=data.get("training_job_id"),  # must set explicitly;
                sampling_job_id=data.get("sampling_job_id"),  # Field(exclude=True)
                log_prob_job_id=data.get("log_prob_job_id"),  # skips json serde
            )
            state["tokenizer"] = AutoTokenizer.from_pretrained(reconnect_cfg.model_name)
            state["client"] = ArcticRLClient(reconnect_cfg)  # reconnect, no /initialize
            state["ready"] = True

        asyncio.create_task(asyncio.get_event_loop().run_in_executor(None, _blocking_init))
        yield
```

Two design choices here:

1. **`surface_model` read eagerly** — the orchestrator calls `/v1/models`
   right after `/health 200`, before `_blocking_init` finishes. Reading
   the model name synchronously at startup (fast, tiny JSON file) ensures
   the response is always correct.

2. **`ArcticRLClientConfig(**data)` not `model_validate_json()`** —
   `training_job_id` and `sampling_job_id` are `Field(exclude=True)` in
   Pydantic, so `model_validate_json` silently leaves them `None`, causing
   `ArcticRLClient` to call `/initialize` again. Building the config from
   the dict directly preserves the IDs.

3. **Launcher waits for `ready: True`** — the fire-and-forget `create_task`
   means `/health` returns `200` immediately, but `state["client"]` is `None`
   until `_blocking_init` finishes (~30s cold). The launcher polls
   `health["ready"] == True` before spawning the orchestrator.

### /v1/chat/completions handler

```python
# src/prime_rl/arctic/shim/app.py:98 chat_completions
@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    sampling = oai_sampling_params(body)     # translation.py:14

    # Render OAI chat_template → string
    prompts = [app.state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )]
    # (Offload the blocking HTTP call to a thread so asyncio event loop stays free.)
    results = await loop.run_in_executor(None,
        lambda: app.state.client.generate(prompts, sampling))   # POST /generate
    return dss_results_to_oai_response(results, ...)            # translation.py:72
```

`oai_sampling_params` maps OpenAI fields (`temperature`, `top_p`,
`max_tokens`, `n`, `stop`, `logprobs`, `seed`) to DSS's `sampling_params`
dict. `dss_results_to_oai_response` reshapes the DSS
`{"results": [{text, token_ids, finish_reason, logprobs}, ...]}` payload
into OAI `{"id", "object", "choices": [{"index", "message", ...}]}`.

Other routes — `/pause`, `/resume`, `/update_weights`, `/init_broadcaster`
— are all no-op acks. Prime-RL's orchestrator calls them (thinking it's
vLLM) but DSS handles everything server-side via `/sync-weights`.

## orchestrator (subprocess 3)

This is vanilla Prime-RL. Key interaction surface with Arctic:

1. **Rollout generation** hits `config.orchestrator.client.base_url` which
   the launcher rewrote to point at the shim. Each `/v1/chat/completions`
   call eventually produces one or more rollouts.

2. **Rollout write** — after collecting `batch_size` rollouts (128 for
   reverse-text), orchestrator writes
   `outputs/run_default/rollouts/step_N/train_rollouts.bin` (serialized
   `TrainingBatch`). This is the handoff to the trainer.

3. **STABLE gate** — before starting step N+1 rollouts, orchestrator polls
   `outputs/run_default/broadcasts/step_N/STABLE`. Trainer touches that
   file after `/sync-weights`.

## DSS server (ports 7000 + 8000+)

Router: `dss-platform/dss/zone_server.py` (FastAPI on port 7000).

| Endpoint | Line | Purpose |
|---|---|---|
| POST `/register-device` | 150 | device manager registers on startup |
| POST `/initialize` | 186 | create training or sampling job |
| POST `/fwd-bwd` | 273 | binary request → DeepSpeed fwd+bwd, JSON response |
| POST `/step` | 302 | apply optimizer, return `last_lr` |
| POST `/sync-weights` | 345 | NCCL transfer training → sampling weights |
| POST `/generate` | 424 | vLLM rollout sampling |
| POST `/destroy` | 247 | tear down a job |
| GET `/status` | 467 | list jobs |

Each job type has a manager: `TrainingJobManager`, `SamplingJobManager`,
`LogProbJobManager`. Devices register with the appropriate manager at
startup (training zone fills first; remainder go to sampling). When all
devices in a zone are registered, the manager initializes its runtime:
vLLM `Driver` for sampling, DeepSpeed `deepspeed.initialize()` for
training.

## Full timeline of one training step

```
t=0     orchestrator: collect 128 rollouts via /v1/chat/completions
                      ↓ (each call hits shim → DSS /generate → vLLM)
t=~85s  orchestrator: compute advantages, drop zero-advantage rollouts
t=~85s  orchestrator: write outputs/run_default/rollouts/step_N/train_rollouts.bin
                      ↓
t=~85s  Prime-RL packer: reads train_rollouts.bin, packs into [1, T] bins,
        writes outputs/rollouts/step_N/rank_0.bin, sets ready_to_update[0]=True
                      ↓
        arctic-trainer DataLoader: reads rank_0.bin → mbs = [TensorMicroBatch, ...]
                      ↓
        trainer: microbatches_to_arctic_context(mbs) → one [B=128, max_S] tensor
                      ↓
        trainer: POST /fwd-bwd?job_id=<training_job_id> (binary torch.save payload)
                      ↓
        DSS zone:  dispatch to training engine on training-zone devices
                      ↓
        DSS training engine: torch.chunk(dim=0, world_size) → shard across DP ranks
                             each rank: pack_sequences → DeepSpeed fwd/bwd
                             return JSON {avg_loss, metrics}
                      ↓
        trainer: POST /step → DeepSpeed optimizer.step() → {last_lr, grad_norm}
                      ↓
        trainer (if step > 0): POST /sync-weights
                      ↓
        DSS zone: orchestrate NCCL transfer training rank 0 → sampling replicas
                  (pair-wise comms, one per replica)
                      ↓
        trainer (if step > 0): touch outputs/run_default/broadcasts/step_N/STABLE
                      ↓
        trainer: ready_to_update[0] = False so next pack() can run
                      ↓
        orchestrator: detects step_N/STABLE, proceeds with step N+1 rollouts
```

## Configuration knobs

| Config key | Where | What it affects |
|---|---|---|
| `arctic.backend = "dss"` | `rl.toml` | enables `rl_arctic_local` dispatch |
| `arctic.url` | `rl.toml` | DSS zone URL (client-side only) |
| `arctic.training_gpus` | `rl.toml` | requested training GPUs; forwarded as `training_config.n_gpus` |
| `arctic.sampling_tensor_parallel_size` | `rl.toml` | vLLM TP size per sampling replica |
| `arctic.vllm_config.max_model_len` | `rl.toml` | sampling context length; defaults to `trainer.model.seq_len` |
| `arctic.shim_host` / `shim_port` | `rl.toml` | shim bind addr; `shim_port=0` picks free |
| `max_steps` | `rl.toml` | how many training steps before exit |
| `--training-zone-size` | `dss-zone` CLI | DP width on training side (we tested up to 2) |
| `--sampling-zone-size` | `dss-zone` CLI | sampling zone GPUs; replicas = zone size / `sampling_tensor_parallel_size` |
| `ARCTIC_SHIM=1` | launcher env | skips `prime_rl._compat` for shim boot |
| `ARCTIC_CONFIG_TOML` | launcher env | points arctic-trainer at the sidecar arctic.toml |
| `outputs/configs/reconnect.json` | file (trainer writes, launcher/shim reads) | full reconnect state: host, port, model_name, all three job IDs |

## Key files, by role

| File | Role |
|---|---|
| `src/prime_rl/entrypoints/rl.py:407 rl_arctic_local` | supervises the 3 subprocesses |
| `src/prime_rl/arctic/entrypoint.py:43 main` | arctic-trainer entry |
| `src/prime_rl/arctic/trainer.py:87 ArcticTrainerAdapter` | HTTP-driven training loop |
| `src/prime_rl/arctic/client.py:56 build_arctic_client` | builds `ArcticRLClient` |
| `src/prime_rl/arctic/context.py:70 microbatches_to_arctic_context` | unpack+roll+pad to `[B, max_S]` |
| `src/prime_rl/arctic/unpack.py:38 _detect_rollout_starts` | rollout boundary detector |
| `src/prime_rl/arctic/shim/entrypoint.py` | arctic-shim entry |
| `src/prime_rl/arctic/shim/app.py:12 create_app` | FastAPI app factory |
| `src/prime_rl/arctic/shim/translation.py` | OAI↔DSS payload shape |
| `ArcticTraining-dss/arctic_training/arctic_rl/client.py:56 ArcticRLClient` | HTTP client lib |
| `dss-platform/dss/zone_server.py` | DSS FastAPI router |
| `dss-platform/dss/job_manager/training.py` | routes `/fwd-bwd`, `/step`, etc. |
| `dss-platform/dss/job_manager/sampling.py` | routes `/generate` via `ArcticInference.Driver` |
| `dss-platform/dss/engines/gpu/training.py` | `TrainingJobEngine` (DeepSpeed) |
| `dss-platform/dss/engines/gpu/sampling.py` | `AsyncSamplingJobEngine` (vLLM) |

## Boot time budget (observed on 8×H200, cold cache)

| Phase | Duration | Notes |
|---|---|---|
| Launcher CLI parse | ~1s | tyro + model pre-download check |
| arctic-trainer imports | ~77s | prime_rl + DeepSpeed + transformers |
| DSS `/initialize` sampling (6× vLLM) | ~62s | model load in parallel |
| DSS `/initialize` training (DeepSpeed) | ~18s | weights HF download + DS init |
| arctic-shim boot | ~4s | ARCTIC_SHIM=1 skips `_compat` |
| orchestrator imports + env ready | ~2m 52s | prime_rl + verifiers + wandb |
| **Total to first step 0 rollout** | **~6–7 min** | |
| Step 0 (cold vLLM) | ~85s | |
| Steady-state step (2T+6S, 128 rollouts) | ~1.1s | generation overlapped with training |

## Related PRs

| PR | What it fixed |
|---|---|
| [#5](https://github.com/snowflake-eng/arctic-primerl/pull/5) | Unpack packed `[1, T]` → `[B, S]` for DSS DP>1 |
| [#7](https://github.com/snowflake-eng/arctic-primerl/pull/7) | Clear `ready_to_update` so packer unblocks past step 0 |
| [#8](https://github.com/snowflake-eng/arctic-primerl/pull/8) | Consolidate all microbatches into one `/fwd-bwd` per step |
| [#9](https://github.com/snowflake-eng/arctic-primerl/pull/9) | Replace `ARCTIC_SAMPLING_JOB_ID` stdout marker with `reconnect.json` file |
