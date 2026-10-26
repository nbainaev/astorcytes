"""Command line entry point for training, tuning, scientific checks and plotting."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "tune", "plot", "check"])
    parser.add_argument(
        "--workers", type=int, default=30, help="Parallel training workers (1–30; default: 30)."
    )
    parser.add_argument(
        "--output", help="Directory for generated metrics, checkpoints and figures."
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 30:
        parser.error("--workers must be between 1 and 30")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    if args.output:
        os.environ["ASTRORL_OUTPUT_DIR"] = args.output
    # Numerical packages are imported after their thread budgets are configured.
    if args.command in ("run", "tune"):
        from . import runner

        getattr(runner, args.command)(workers=args.workers)
    elif args.command == "plot":
        from .analysis import report

        report()
    else:
        from .checks import main as check

        check()


if __name__ == "__main__":
    main()
