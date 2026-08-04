"""E8B: E8A와 일반 5-Fold EM16 확률의 cross-fit 앙상블."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ensembles.common import run_em16_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ensembles/test_006_v01.yaml"),
    )
    args = parser.parse_args()
    run_em16_ensemble(args.config)


if __name__ == "__main__":
    main()
