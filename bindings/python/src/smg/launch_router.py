from __future__ import annotations

import argparse
import logging
import sys

import setproctitle

from smg.router_args import RouterArgs
from smg.smg_rs import print_banner

logger = logging.getLogger("router")

try:
    from smg.router import Router
except ImportError:
    Router = None  # type: ignore[assignment,misc]
    logger.warning("Rust Router is not installed")


# ---------------------------------------------------------------------------
# K3 KVV overlay — strictly opt-in.
#
# Activates ONLY when K3_KVV_FACADE_ENABLED=1 is set (e.g. in /spinfield/env).
# The unmodified Rust gateway then moves to a private loopback port and the
# k3_kvv_facade protocol facade owns the public one. Default: OFF.
# ---------------------------------------------------------------------------
def _k3_kvv_upstream_port() -> int:
    import os

    return int(os.environ.get("K3_KVV_UPSTREAM_PORT", "8300"))


_K3_KVV_FACADE_PROC = None
_K3_KVV_UPSTREAM_HOST = "127.0.0.1"
_K3_KVV_UPSTREAM_PORT = _k3_kvv_upstream_port()

def _k3_kvv_should_overlay() -> bool:
    """Strictly opt-in: the overlay activates ONLY when the operator sets
    K3_KVV_FACADE_ENABLED=1 (e.g. in /spinfield/env). Default in the repo is
    OFF so upstream behavior is byte-identical without explicit opt-in."""
    import os

    if os.environ.get("K3_KVV_FACADE_ENABLED") != "1":
        return False
    try:
        import smg.k3_kvv_facade  # noqa: F401
    except ImportError:
        return False
    return True


def _k3_kvv_start_facade(public_host: str, public_port: int | str) -> None:
    """Start the facade subprocess bound to the public address before the
    Rust gateway takes the loopback port; probe failures while the facade
    is up therefore reflect the real backend, never a false healthy."""
    import os
    import subprocess
    import sys
    import time

    global _K3_KVV_FACADE_PROC
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "K3_KVV_UPSTREAM_PORT": str(_k3_kvv_upstream_port()),
        "K3_KVV_FACADE_PORT": str(public_port),
    }
    _K3_KVV_FACADE_PROC = subprocess.Popen(
        [sys.executable, "-m", "smg.k3_kvv_facade"],
        env=env,
    )
    time.sleep(1.0)
    if _K3_KVV_FACADE_PROC.poll() is not None:
        raise RuntimeError(
            f"k3_kvv_facade exited immediately with code {_K3_KVV_FACADE_PROC.returncode}"
        )
    logger.info(
        "k3-kvv facade on %s:%s -> smg on %s:%s",
        public_host, public_port, _K3_KVV_UPSTREAM_HOST, env["K3_KVV_UPSTREAM_PORT"],
    )


def launch_router(args: argparse.Namespace | RouterArgs) -> None:
    """
    Launch the SMG router with the configuration from parsed arguments.

    Args:
        args: Namespace object containing router configuration
            Can be either raw argparse.Namespace or converted RouterArgs

    Returns:
        None. Raises on failure.
    """
    setproctitle.setproctitle("smg::router")
    try:
        # Convert to RouterArgs if needed
        if not isinstance(args, RouterArgs):
            router_args = RouterArgs.from_cli_args(args)
        else:
            router_args = args

        if Router is None:
            raise RuntimeError("Rust Router is not installed")

        if router_args.selected_discovery_provider() and not router_args.enable_igw:
            logger.info("IGW mode automatically enabled because service discovery is turned on")
            router_args.enable_igw = True

        router_args._validate_router_args()

        # Determine mode for banner
        if getattr(router_args, "enable_igw", False):
            mode = "IGW"
        elif getattr(router_args, "pd_disaggregation", False):
            mode = "PD Disaggregated"
        else:
            mode = "Regular"
        public_host, public_port = router_args.host, router_args.port
        if _k3_kvv_should_overlay():
            # The facade owns the public port; the Rust gateway moves to the
            # private loopback one and keeps its original startup args intact
            # apart from the bind address.
            router_args.host = _K3_KVV_UPSTREAM_HOST
            router_args.port = _K3_KVV_UPSTREAM_PORT
        print_banner(router_args.host, router_args.port, mode)
        if _k3_kvv_should_overlay():
            _k3_kvv_start_facade(public_host, public_port)

        router = Router.from_args(router_args)
        router.start()

    except Exception as e:
        logger.error(f"Error starting router: {e}")
        raise e


class CustomHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter that preserves both description formatting and shows defaults"""

    pass


def parse_router_args(args: list[str]) -> RouterArgs:
    """Parse command line arguments and return RouterArgs instance."""
    parser = argparse.ArgumentParser(
        description="""SMG Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  python -m smg.launch_router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --policy cache_aware

  # PD mode with optional bootstrap ports
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 \\    # With bootstrap port
    --prefill http://prefill2:8000 none \\    # Explicitly no bootstrap port
    --prefill http://prefill3:8000 \\         # Defaults to no bootstrap port
    --decode http://decode1:8001 --decode http://decode2:8001

  # PD mode with different policies for prefill and decode
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --prefill-policy cache_aware --decode-policy power_of_two

  # EPD disaggregated mode
  python -m smg.launch_router --epd-disaggregation \\
    --encode http://encode1:8000 9000 \\
    --prefill http://prefill1:8000 9001 \\
    --decode http://decode1:8001 \\
    --encode-policy consistent_hashing

    """,
        formatter_class=CustomHelpFormatter,
    )

    RouterArgs.add_cli_args(parser, use_router_prefix=False)
    return RouterArgs.from_cli_args(parser.parse_args(args), use_router_prefix=False)


def main() -> None:
    router_args = parse_router_args(sys.argv[1:])
    launch_router(router_args)


if __name__ == "__main__":
    main()
