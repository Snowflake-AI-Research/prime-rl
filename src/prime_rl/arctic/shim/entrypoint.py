"""arctic-shim entrypoint.

Launches the FastAPI OAI-compat app via uvicorn. The launcher
(rl_arctic_local) spawns this subprocess after arctic-trainer has written
outputs/configs/reconnect.json. The shim loads that config and passes it
to ArcticRLClient, which reconnects to the existing DSS sampling job
without calling /initialize.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from prime_rl.arctic.shim.app import create_app
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("ArcticShim")
    parser = argparse.ArgumentParser(description="OpenAI-compat shim for DSS /generate")
    parser.add_argument(
        "--reconnect-config",
        required=True,
        help="Path to reconnect.json written by arctic-trainer (contains host, port, job IDs)",
    )
    parser.add_argument("--model-name", default=None, help="Model name for /v1/models (defaults to config model_name)")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to tokenizer.apply_chat_template.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--fd",
        type=int,
        default=None,
        help=(
            "Inherited file descriptor of a pre-bound socket. When set, uvicorn "
            "listens on this fd instead of binding --host/--port itself. Used by "
            "the launcher (rl_arctic_local) to avoid a TOCTOU race between port "
            "allocation and the shim's bind."
        ),
    )
    args = parser.parse_args()

    reconnect_path = Path(args.reconnect_config)
    if not reconnect_path.exists():
        raise FileNotFoundError(f"reconnect.json not found: {reconnect_path}")

    app = create_app(
        reconnect_config_path=reconnect_path,
        model_name=args.model_name,
        enable_thinking=args.enable_thinking,
    )
    if args.fd is not None:
        logger.info(
            "Starting arctic-shim on fd={} (nominal {}:{}, reconnect_config={})",
            args.fd,
            args.host,
            args.port,
            reconnect_path,
        )
        uvicorn.run(app, fd=args.fd, log_level="info")
    else:
        logger.info(
            "Starting arctic-shim on {}:{} (reconnect_config={})",
            args.host,
            args.port,
            reconnect_path,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
