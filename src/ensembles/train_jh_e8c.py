"""E8C: E8A와 동일 SGKF EM16 확률의 aligned cross-fit 앙상블."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ensembles.common import run_em16_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ensembles/test_006_v02.yaml"),
    )
    args = parser.parse_args()
    run_em16_ensemble(args.config)


if __name__ == "__main__":
    main()
