import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import pynvml
import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    format_log_message,
    get_ckpt_dir,
    get_log_dir,
    resolve_latest_ckpt_step,
    validate_output_dir,
)
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title

RL_TOML = "rl.toml"
RL_SBATCH = "rl.sbatch"

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
TEACHER_INFERENCE_TOML = "teacher_inference.toml"
ARCTIC_TOML = "arctic.toml"


def get_physical_gpu_ids() -> list[int]:
    """Return physical GPU IDs visible to the launcher."""
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        pynvml.nvmlInit()
        return list(range(pynvml.nvmlDeviceGetCount()))
    return [int(token.strip()) for token in raw_visible.split(",") if token.strip()]


def write_config(config: RLConfig, output_dir: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(exclude=exclude, exclude_none=True, mode="json")
    with open(output_dir / RL_TOML, "wb") as f:
        tomli_w.dump(config_dict, f)


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / TRAINER_TOML, "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / ORCHESTRATOR_TOML, "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        # Exclude launcher-only fields that are not needed by the vLLM server
        exclude_inference = {"deployment", "slurm", "output_dir", "dry_run"}
        with open(output_dir / INFERENCE_TOML, "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude=exclude_inference, exclude_none=True, mode="json"), f)

    teacher_inference = getattr(config, "teacher_inference", None)
    if teacher_inference is not None:
        with open(output_dir / TEACHER_INFERENCE_TOML, "wb") as f:
            tomli_w.dump(teacher_inference.model_dump(exclude_none=True, mode="json"), f)


def rl_local(config: RLConfig):
    if config.arctic is not None and config.arctic.backend is not None:
        return rl_arctic_local(config)

    assert config.deployment.type == "single_node"

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an RL run locally, remove --dry-run from your command.")
        return

    # Derive launcher-local GPU IDs from deployment config
    gpu_offset = 0
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    infer_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_infer_gpus))
    gpu_offset += num_infer_gpus
    trainer_local_gpu_ids = list(range(gpu_offset, gpu_offset + config.deployment.num_train_gpus))
    gpu_offset += config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0
    teacher_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_teacher_gpus)) if num_teacher_gpus > 0 else []

    total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus + num_teacher_gpus
    physical_gpu_ids = get_physical_gpu_ids()
    if total_requested_gpus > len(physical_gpu_ids):
        raise ValueError(
            f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
            f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
        )
    physical_gpu_mapping = {local_id: physical_gpu_ids[local_id] for local_id in range(total_requested_gpus)}
    logger.info(f"Using local->physical GPU mapping: {physical_gpu_mapping}")

    infer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in infer_local_gpu_ids]
    trainer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in trainer_local_gpu_ids]
    teacher_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in teacher_local_gpu_ids]

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Build shared W&B env vars for subprocesses
    wandb_shared_env: dict[str, str] = {}
    if config.wandb and config.wandb.shared:
        wandb_shared_env["WANDB_SHARED_MODE"] = "1"
        wandb_shared_env["WANDB_SHARED_RUN_ID"] = os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex)

    # Validate client port matches inference server port
    if config.inference is not None and not config.orchestrator.student.client.is_elastic:
        from urllib.parse import urlparse

        base_url = config.orchestrator.student.client.base_url[0]
        parsed = urlparse(base_url)
        client_port = parsed.port
        expected_port = config.inference.server.port
        if client_port != expected_port:
            raise ValueError(
                f"orchestrator.student.client.base_url port ({client_port}) does not match "
                f"inference.server.port ({expected_port}). "
                f"Update the base_url to use port {expected_port} to match the inference server."
            )

    # Prepare paths to communicate with the trainer
    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # Optionally, start inference process
        if config.inference:
            inference_cmd = ["inference", "@", (config_dir / INFERENCE_TOML).as_posix()]
            logger.info(f"Starting inference on GPU(s) {' '.join(map(str, infer_gpu_ids))}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            # If we don't log stdout, the server hangs
            with open(log_dir / "inference.log", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, infer_gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No [inference] block configured - the student inference server will not be started here. "
                "All training modes (rl/opd/sft) require a student inference pool for evals + weight sync; "
                "make sure one is running at orchestrator.student.client.base_url "
                f"({', '.join(config.orchestrator.student.client.base_url)}), otherwise the orchestrator "
                "will hang waiting for it."
            )

        # Optionally, start teacher inference process
        if config.teacher_inference:
            if not teacher_gpu_ids:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set. "
                    "Either set deployment.num_teacher_gpus to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher to use an existing server."
                )

            teacher_inference_cmd = ["inference", "@", (config_dir / TEACHER_INFERENCE_TOML).as_posix()]
            logger.info(f"Starting teacher inference process on GPU(s) {' '.join(map(str, teacher_gpu_ids))}")
            logger.debug(f"Teacher inference start command: {' '.join(teacher_inference_cmd)}")
            with open(log_dir / "teacher_inference.log", "w") as log_file:
                teacher_inference_process = Popen(
                    teacher_inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, teacher_gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(teacher_inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["teacher_inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(teacher_inference_process, stop_event, error_queue, "teacher_inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        elif config.orchestrator.teacher:
            logger.warning(
                "No teacher_inference config specified, skipping starting teacher inference server. "
                "Is your teacher inference server running? Make sure orchestrator.teacher is configured."
            )

        # Start orchestrator process
        orchestrator_cmd = [
            "orchestrator",
            "@",
            (config_dir / ORCHESTRATOR_TOML).as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(log_dir / "orchestrator.log", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "orchestrator",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Start training process
        from prime_rl.utils.utils import get_free_port

        trainer_cmd = [
            "torchrun",
            "--role=trainer",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
            f"--local-ranks-filter={','.join(map(str, config.trainer.log.ranks_filter))}",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (config_dir / TRAINER_TOML).as_posix(),
        ]
        logger.info(f"Starting trainer on GPU(s) {' '.join(map(str, trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.log", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "trainer",
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, trainer_gpu_ids)),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process, args=(trainer_process, stop_event, error_queue, "trainer"), daemon=True
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Monitor all processes for failures
        logger.success("Startup complete. Showing orchestrator logs...")

        tail_process = Popen(
            f"tail -F '{log_dir / 'orchestrator.log'}'",
            shell=True,
        )
        processes.append(tail_process)

        # Check for errors from monitor threads
        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)

            # Small delay to avoid busy waiting
            time.sleep(1)

        # Check if any critical process failed
        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("RL training finished!")

        # Cleanup threads and processes
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise


def write_slurm_script(config: RLConfig, config_dir: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_dir / RL_TOML,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    elif config.inference is not None and config.inference.deployment.type == "disaggregated":
        infer_deploy = config.inference.deployment

        script = template.render(
            **config.slurm.template_vars,
            is_disaggregated=True,
            config_dir=config_dir,
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=infer_deploy.num_nodes * config.deployment.num_infer_replicas,
            nodes_per_infer_replica=infer_deploy.num_nodes,
            num_infer_replicas=config.deployment.num_infer_replicas,
            num_prefill_nodes=infer_deploy.num_prefill_nodes,
            num_decode_nodes=infer_deploy.num_decode_nodes,
            num_prefill_replicas=infer_deploy.num_prefill_replicas,
            num_decode_replicas=infer_deploy.num_decode_replicas,
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=infer_deploy.router_port,
            prefill_port=infer_deploy.prefill_port,
            decode_port=infer_deploy.decode_port,
            inference_tp=config.inference.parallel.tp,
            inference_data_parallel_rpc_port=config.inference.data_parallel_rpc_port,
            use_deep_gemm=config.inference.use_deep_gemm,
            prefill_env_overrides=infer_deploy.prefill_env_overrides,
            decode_env_overrides=infer_deploy.decode_env_overrides,
            dp_per_node=config.deployment.gpus_per_node // config.inference.parallel.tp,
            kv_offload=config.inference.kv_cache_offload is not None,
            kv_offload_cpu_bytes=int(config.inference.kv_cache_offload.cpu_bytes)
            if config.inference.kv_cache_offload
            else 0,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
            ranks_filter=",".join(map(str, config.trainer.log.ranks_filter)),
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            is_disaggregated=False,
            config_dir=config_dir,  # TODO: should prob have each subconfig path separately
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.total_infer_nodes,
            nodes_per_infer_replica=config.deployment.num_infer_nodes,
            num_infer_replicas=config.deployment.num_infer_replicas,
            num_teacher_nodes=config.deployment.num_teacher_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            router_port=getattr(config.inference.deployment, "router_port", 8000) if config.inference else 8000,
            backend_port=getattr(config.inference.deployment, "backend_port", 8100) if config.inference else 8100,
            inference_tp=config.inference.parallel.tp if config.inference else 1,
            inference_enable_expert_parallel=config.inference.enable_expert_parallel if config.inference else False,
            inference_data_parallel_rpc_port=config.inference.data_parallel_rpc_port if config.inference else 29600,
            dp_per_node=(config.deployment.gpus_per_node // config.inference.parallel.tp) if config.inference else 1,
            kv_offload=config.inference is not None and config.inference.kv_cache_offload is not None,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
            wandb_shared=config.wandb is not None and config.wandb.shared,
            ranks_filter=",".join(map(str, config.trainer.log.ranks_filter)),
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def rl_slurm(config: RLConfig):
    assert config.slurm is not None

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"), json_logging=config.log.json_logging
    )

    config_dir = config.output_dir / "configs"
    log_dir = get_log_dir(config.output_dir)

    if config.deployment.type == "single_node":
        write_config(config, config_dir, exclude={"slurm", "dry_run", "clean_output_dir"})
        logger.info(f"Wrote config to {config_dir / RL_TOML}")

        train_env_names = [env.resolved_name for env in config.orchestrator.train.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        log_message = format_log_message(
            log_dir=log_dir,
            trainer=True,
            orchestrator=True,
            inference=True,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
        )
    else:
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")

        train_env_names = [env.resolved_name for env in config.orchestrator.train.env]
        eval_env_names = [env.resolved_name for env in config.orchestrator.eval.env] if config.orchestrator.eval else []

        has_infer = config.deployment.num_infer_nodes > 0
        log_message = format_log_message(
            log_dir=log_dir,
            trainer=True,
            orchestrator=has_infer,
            inference=has_infer,
            train_env_names=train_env_names,
            eval_env_names=eval_env_names,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.total_infer_nodes if has_infer else 0,
        )

    script_path = config.output_dir / RL_SBATCH
    write_slurm_script(config, config_dir, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def rl(config: RLConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean, ckpt_output_dir=ckpt_output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_output_dir is not None:
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale rollouts and broadcasts. When resuming, anything past the resume
    # step is stale. When training from scratch, every existing step directory is
    # stale — without this, a fresh run in a dirty output_dir would pick up rollouts
    # from a previous run and the orchestrator would see a negative async level.
    resume_step: int | None = None
    if resuming:
        resume_step = config.ckpt.resume_step
        if resume_step == -1:
            ckpt_base = ckpt_output_dir if ckpt_output_dir is not None else config.output_dir
            resume_step = resolve_latest_ckpt_step(get_ckpt_dir(ckpt_base))

    if resume_step is not None:
        get_logger().info(f"Resuming from step {resume_step}, cleaning future rollouts and broadcasts")
        clean_future_steps(config.output_dir, resume_step)
    else:
        get_logger().info("Training from scratch, cleaning any stale rollouts and broadcasts")
        clean_future_steps(config.output_dir, -1)

    if not config.dry_run:
        from prime_rl.trainer.model import pre_download_model

        pre_download_model(config.trainer.model.name)

    if config.slurm is not None:
        rl_slurm(config)
    else:
        rl_local(config)


def main():
    set_proc_title("Launcher")
    rl(cli(RLConfig))


if __name__ == "__main__":
    main()
def rl_arctic_local(config: RLConfig):
    """Arctic-mode launcher: arctic-trainer + arctic-shim + orchestrator.

    Replaces the native path's inference + torchrun trainer subprocesses with:
      1. `arctic-trainer` (single-process; wraps ArcticRLClient over HTTP)
      2. `arctic-shim` (FastAPI OAI-compat proxy to DSS /generate)
      3. Orchestrator (unchanged, just pointed at the shim via base_url)

    The trainer writes a reconnect.json to outputs/configs/ once its DSS jobs
    are RUNNING; the launcher detects the file and passes its path to the shim
    so the shim can call ArcticRLClient(reconnect_cfg) without /initialize.
    """
    import http.client
    import urllib.error
    import urllib.request

    assert config.arctic is not None and config.arctic.backend is not None
    arctic_cfg = config.arctic

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    # Allocate a free port dynamically so concurrent arctic runs don't conflict.
    # Using a fixed port (e.g. 8010) would cause "address already in use" errors
    # on machines running multiple experiments simultaneously.
    #
    # We bind the socket here and pass the fd to the shim via pass_fds + --fd,
    # rather than allocating a port with get_free_port() and letting the shim
    # bind it later. Holding the socket open eliminates a TOCTOU race where
    # another process (typically a Ray worker spawning during sampling-engine
    # init) grabs the same ephemeral port between allocation and the shim's
    # actual bind. That race manifests as the shim exiting with EADDRINUSE,
    # and the launcher's health-poll receiving garbage from whoever squatted
    # the port.
    from prime_rl.utils.utils import reserve_free_port

    shim_sock = reserve_free_port()
    shim_port = shim_sock.getsockname()[1]
    shim_base_url = f"http://{arctic_cfg.shim_host}:{shim_port}/v1"
    config.orchestrator.client.base_url = [shim_base_url]

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    arctic_toml_path = config_dir / ARCTIC_TOML
    arctic_toml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(arctic_toml_path, "wb") as f:
        tomli_w.dump(arctic_cfg.model_dump(exclude_none=True, mode="json"), f)
    logger.info(f"Wrote subconfigs (including arctic.toml) to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete (arctic mode).")
        return

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating arctic processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # 1. Start trainer. Redirect stdout+stderr to the log file directly;
        #    no PIPE needed — trainer signals readiness via reconnect.json.
        # Delete any stale reconnect.json from a prior run FIRST. Otherwise the
        # launcher sees old job_ids / old zone URLs and boots the shims against
        # ghosts while the trainer is still /initializing (or has failed).
        (config_dir / "reconnect.json").unlink(missing_ok=True)
        (config_dir / "judge_reconnect.json").unlink(missing_ok=True)

        trainer_log_path = log_dir / "arctic_trainer.log"
        trainer_cmd = [
            "arctic-trainer",
            "@",
            (config_dir / TRAINER_TOML).as_posix(),
        ]
        logger.info("Starting arctic-trainer (%s)", " ".join(trainer_cmd))

        trainer_env = {
            **os.environ,
            "ARCTIC_CONFIG_TOML": arctic_toml_path.as_posix(),
            # Arctic trainer is single-process, no GPU on the client side.
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "LOGURU_FORCE_COLORS": "1",
        }

        with open(trainer_log_path, "w") as trainer_log_file:
            trainer_process = Popen(
                trainer_cmd,
                env=trainer_env,
                stdout=trainer_log_file,
                stderr=trainer_log_file,
            )
        processes.append(trainer_process)

        # Wait for reconnect.json, written by the trainer once DSS jobs are RUNNING.
        # No fixed timeout: raise immediately if the trainer process exits; otherwise
        # keep waiting. DSS's own job_timeout_seconds acts as the runaway guard.
        reconnect_path = config_dir / "reconnect.json"
        logger.info("Waiting for arctic-trainer to initialize DSS jobs…")
        while not reconnect_path.exists():
            if trainer_process.poll() is not None:
                raise RuntimeError(
                    f"arctic-trainer exited (code {trainer_process.returncode}) "
                    f"before writing reconnect.json. See {trainer_log_path}."
                )
            time.sleep(1)
        logger.success("arctic-trainer ready (reconnect.json written)")

        stop_event = Event()
        stop_events["arctic_trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "arctic_trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # 2. Start shim. Pass the reconnect config path so it can attach to
        #    the sampling job without calling /initialize. We pass the bound
        #    socket fd directly (see reserve_free_port above); the child will
        #    use it instead of binding --host/--port itself. --host/--port are
        #    still passed as metadata (for logging) but --fd takes precedence.
        shim_cmd = [
            "arctic-shim",
            "--reconnect-config",
            reconnect_path.as_posix(),
            "--host",
            arctic_cfg.shim_host,
            "--port",
            str(shim_port),
            "--fd",
            str(shim_sock.fileno()),
            "--model-name",
            config.trainer.model.name,
        ]
        if arctic_cfg.enable_thinking:
            shim_cmd.append("--enable-thinking")
        shim_log_path = log_dir / "arctic_shim.log"
        logger.info("Starting arctic-shim on %s:%d (fd=%d)", arctic_cfg.shim_host, shim_port, shim_sock.fileno())
        shim_env = {**os.environ, "ARCTIC_SHIM": "1"}
        with open(shim_log_path, "w") as shim_log_file:
            shim_process = Popen(
                shim_cmd,
                stdout=shim_log_file,
                stderr=shim_log_file,
                env=shim_env,
                pass_fds=(shim_sock.fileno(),),
            )
        processes.append(shim_process)
        # Release our handle to the socket; the child inherited its own.
        shim_sock.close()

        # Wait for shim /health AND ready:True.
        # The shim's /health returns 200 immediately (fire-and-forget init),
        # but state["client"] is None until _blocking_init completes. The
        # orchestrator's first /v1/chat/completions would race and hit
        # AttributeError: 'NoneType'.generate() if we start it too soon.
        import json as _json

        health_url = f"http://{arctic_cfg.shim_host}:{shim_port}/health"
        # No fixed timeout: cold imports of arctic_training+transformers+
        # deepspeed in the shim's _blocking_init can take 2-6 minutes the first
        # time page cache isn't warmed. Fail-fast only on process exit.
        #
        # Catch http.client.HTTPException too — if something other than our
        # shim happens to be squatting on the polled port (historically this
        # was caused by the get_free_port race now fixed via fd-passing), it
        # may answer with non-HTTP garbage and raise BadStatusLine. Swallow
        # it; the shim_process.poll() check above is the authoritative failure
        # path.
        while True:
            if shim_process.poll() is not None:
                raise RuntimeError(f"arctic-shim exited before becoming healthy. See {shim_log_path}.")
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        data = _json.loads(resp.read())
                        if data.get("ready"):
                            break
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
            ):
                pass
            time.sleep(0.5)
        logger.success("arctic-shim healthy")

        stop_event = Event()
        stop_events["arctic_shim"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(shim_process, stop_event, error_queue, "arctic_shim"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # 2b. Start judge shim if the trainer initialized a judge sampling job.
        reconnect_data = _json.loads(reconnect_path.read_text())
        if reconnect_data.get("judge_sampling_job_id") and arctic_cfg.judge_model_name:
            judge_model_name = arctic_cfg.judge_model_name
            judge_shim_port = arctic_cfg.judge_shim_port

            # Judge zone may be a different DSS on another node. Fall back to
            # the rollout zone's host/port when judge_host/judge_port are absent
            # (co-hosted judge, the default).
            judge_host = reconnect_data.get("judge_host") or reconnect_data["host"]
            judge_port = reconnect_data.get("judge_port") or reconnect_data["port"]

            # Write a separate reconnect.json for the judge shim.
            # training_job_id=-1 is a non-None sentinel so ArcticRLClient
            # enters reconnect mode without calling /initialize.
            judge_reconnect_path = config_dir / "judge_reconnect.json"
            judge_reconnect_path.write_text(
                _json.dumps(
                    {
                        "host": judge_host,
                        "port": judge_port,
                        "backend": reconnect_data["backend"],
                        "model_name": judge_model_name,
                        "training_job_id": -1,
                        "sampling_job_id": reconnect_data["judge_sampling_job_id"],
                        "log_prob_job_id": None,
                    }
                )
            )

            judge_shim_cmd = [
                "arctic-shim",
                "--reconnect-config",
                judge_reconnect_path.as_posix(),
                "--host",
                arctic_cfg.shim_host,
                "--port",
                str(judge_shim_port),
                "--model-name",
                judge_model_name,
            ]
            if arctic_cfg.enable_thinking:
                judge_shim_cmd.append("--enable-thinking")
            judge_shim_log_path = log_dir / "arctic_judge_shim.log"
            logger.info("Starting judge-shim on %s:%d", arctic_cfg.shim_host, judge_shim_port)
            with open(judge_shim_log_path, "w") as judge_shim_log_file:
                judge_shim_process = Popen(
                    judge_shim_cmd, stdout=judge_shim_log_file, stderr=judge_shim_log_file, env=shim_env
                )
            processes.append(judge_shim_process)

            judge_health_url = f"http://{arctic_cfg.shim_host}:{judge_shim_port}/health"
            # No fixed timeout — same cold-import concern as the rollout shim.
            while True:
                if judge_shim_process.poll() is not None:
                    raise RuntimeError(f"judge-shim exited before becoming healthy. See {judge_shim_log_path}.")
                try:
                    with urllib.request.urlopen(judge_health_url, timeout=1) as resp:
                        if resp.status == 200 and _json.loads(resp.read()).get("ready"):
                            break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(0.5)
            logger.success("judge-shim healthy")

            stop_event = Event()
            stop_events["arctic_judge_shim"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(judge_shim_process, stop_event, error_queue, "arctic_judge_shim"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)

        # 3. Start orchestrator (unchanged command).
        orch_cmd = ["orchestrator", "@", (config_dir / ORCHESTRATOR_TOML).as_posix()]
        logger.info("Starting orchestrator")
        with open(log_dir / "orchestrator.log", "w") as orch_log_file:
            orch_process = Popen(
                orch_cmd,
                stdout=orch_log_file,
                stderr=orch_log_file,
                env={
                    **os.environ,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl (arctic)",
                    "WANDB_ARGS": json.dumps(sys.argv),
                },
            )
        processes.append(orch_process)

        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orch_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        logger.success("Arctic startup complete. Tailing trainer log...")
        tail_process = Popen(f"tail -F '{trainer_log_path}'", shell=True)
        processes.append(tail_process)

        while not (stop_events["orchestrator"].is_set() and stop_events["arctic_trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all arctic processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        if orch_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orch_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)
        if trainer_process.returncode != 0:
            logger.error(f"Arctic trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("Arctic RL training finished.")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt, terminating arctic processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise

