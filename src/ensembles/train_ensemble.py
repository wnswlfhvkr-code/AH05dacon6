"""build_and_export_pipeline.py"""

from pathlib import Path
import joblib
import pandas as pd

from src.ensembles.pipeline import build_final_pipeline


def main() -> None:
    # 1. 경로 설정 및 학습 완료된 객체들 불러오기
    models_dir = Path("models")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = joblib.load(models_dir / "jh_v10_generator.pkl")
    baseline_model = joblib.load(models_dir / "e10b1_baseline_model.pkl")
    numeric_model = joblib.load(models_dir / "numeric_lgb_model.pkl")

    # 2. E10B-2 실험 스크립트에서 탐색된 최적 파라미터 적용
    best_temperature = 1.05
    best_numeric_weight = 0.35
    class_names = ["class_0", "class_1", "class_2"]  # LabelEncoder.classes_

    # 3. 파이프라인 묶기
    final_pipeline = build_final_pipeline(
        generator=generator,
        baseline_model=baseline_model,
        numeric_model=numeric_model,
        temperature=best_temperature,
        numeric_weight=best_numeric_weight,
        class_names=class_names,
    )

    # 4. 검증 (Raw DataFrame 입력으로 predict 테스트)
    sample_df = pd.read_csv("data/raw/test.csv").head(5)
    sample_preds = final_pipeline.predict(sample_df)
    print("샘플 예측 성공:", sample_preds)

    # 5. 팀 규격 단일 파이프라인 내보내기 (.pkl)
    save_path = output_dir / "final_ensemble_pipeline.pkl"
    joblib.dump(final_pipeline, save_path)
    print(f"최종 파이프라인 저장 완료: {save_path}")


if __name__ == "__main__":
    main()