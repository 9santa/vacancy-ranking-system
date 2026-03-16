from pathlib import Path
import pandas as pd

from src.models.learned_reranker import LearnedReranker


def main():
    artifact_path = Path("artifacts/learned_reranker.joblib")

    reranker = LearnedReranker(
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512
    )
    reranker.load(artifact_path)

    # Scrab pipeline steps
    pipeline = reranker.model
    clf = pipeline.named_steps["clf"]

    feature_names = reranker.FEATURE_COLUMNS
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": [abs(x) for x in coefficients],
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("\nLearned reranker coefficients:\n")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))

    print(f"\nIntercept: {intercept:.6f}")

    print("\nInterpretation:")
    print("- Positive coefficient -> larger feature value increases relevance probability")
    print("- Negative coefficient -> larger feature value decreases relevance probability")
    print("- Larger absolute value -> stronger influence on the final prediction")


if __name__ == "__main__":
    main()
