import pandas as pd

from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher


JOBS_PATH = "data/raw/jobs_v3_real_clean.csv"
TEST_QUERIES_PATH = "data/raw/test_queries_v4.csv"
MODEL_ARTIFACT_PATH = "artifacts/learned_reranker_v3_5roles.joblib"


def load_queries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["query_id", "resume_text", "target_role_family"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in query file: {missing_columns}")

    if "difficulty" not in df.columns:
        df["difficulty"] = "unknown"

    return df


def hit_at_k(
    recommended_role_families: list[str], target_role_family: str, k: int
) -> int:
    return int(target_role_family in recommended_role_families[:k])


def evaluate_matcher(
    matcher: TwoStageLearnedMatcher,
    eval_df: pd.DataFrame,
    retrieve_top_k: int = 30,
    top_k_values: list[int] = [1, 3, 5],
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for _, row in eval_df.iterrows():
        recommendations = matcher.recommend(
            row["resume_text"],
            retrieve_top_k=retrieve_top_k,
            final_top_k=max(top_k_values),
        )
        predicted_roles = recommendations["role_family"].tolist()

        result_row = {
            "query_id": row["query_id"],
            "difficulty": row["difficulty"],
            "target_role_family": row["target_role_family"],
            "top_1_role": predicted_roles[0] if predicted_roles else None,
        }

        for k in top_k_values:
            result_row[f"hit@{k}"] = hit_at_k(
                predicted_roles, row["target_role_family"], k
            )

        rows.append(result_row)

    results_df = pd.DataFrame(rows)
    metrics = {f"hit@{k}": results_df[f"hit@{k}"].mean() for k in top_k_values}
    by_difficulty = (
        results_df.groupby("difficulty")[[f"hit@{k}" for k in top_k_values]]
        .mean()
        .reset_index()
    )

    return results_df, {"overall": metrics, "by_difficulty": by_difficulty}


def main():
    jobs_df = load_jobs(JOBS_PATH)
    test_df = load_queries(TEST_QUERIES_PATH)

    matcher = TwoStageLearnedMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512,
    )
    matcher.fit_jobs(jobs_df)
    matcher.reranker.load(MODEL_ARTIFACT_PATH)

    results_df, metrics = evaluate_matcher(
        matcher,
        test_df,
        retrieve_top_k=40,
        top_k_values=[1, 3, 5],
    )

    print("\nCurrent model test results:\n")
    print(results_df.to_string(index=False))

    print("\nOverall metrics:\n")
    for metric_name, value in metrics["overall"].items():
        print(f"{metric_name}: {value:.3f}")

    print("\nMetrics by difficulty:\n")
    print(metrics["by_difficulty"].to_string(index=False))


if __name__ == "__main__":
    main()
