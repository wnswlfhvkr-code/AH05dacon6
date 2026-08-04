from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.train_sgkf import write_experiment_report


def test_report_records_train_validation_generalization_gap(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    fold_metrics = pd.DataFrame(
        [
            {
                "seed": 42,
                "fold": 0,
                "feature_count": 10,
                "train_macro_f1": 0.90,
                "validation_macro_f1": 0.40,
                "generalization_gap": 0.50,
                "converged": True,
                "elapsed_seconds": 1.0,
            }
        ]
    )
    seed_metrics = pd.DataFrame([{"seed": 42, "oof_macro_f1": 0.40}])
    config = {
        "project": {"experiment_name": "test_gap"},
        "model": {"name": "linear_svc"},
        "preprocessing": {"name": "jh_v01"},
        "validation": {"n_splits": 1, "seeds": [42]},
    }
    now = datetime.now(timezone.utc)

    write_experiment_report(
        path=report_path,
        config_path=Path("configs/test_gap.yaml"),
        config=config,
        started_at=now,
        finished_at=now,
        train_rows=10,
        fold_metrics=fold_metrics,
        seed_metrics=seed_metrics,
        submission_path=Path("submission.csv"),
        summary_path=Path("summary.json"),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Fold Train Macro F1 평균 | 0.900000" in report
    assert "Fold Validation Macro F1 평균 | 0.400000" in report
    assert "평균 generalization gap | +0.500000" in report
    assert "과적합 경고 Fold (gap > 0.10) | 1/1" in report
