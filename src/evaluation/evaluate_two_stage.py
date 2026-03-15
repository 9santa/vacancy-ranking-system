import pandas as pd

from src.data.load_data import load_jobs
from src.models.two_stage_matcher import TwoStageJobMatcher


def load_eval_queries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["query_id", "resume_text", "target_role_family"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in eval file: {missing_columns}")

    if "difficulty" not in df.columns:
        df["difficulty"] = "unknown"

    return df


def hit_at_k(recommended_role_families: list[str], target_role_family: str, k: int) -> int:
    top_k_roles = recommended_role_families[:k]
    return int(target_role_family in top_k_roles)


def evaluate_matcher(
    matcher: TwoStageJobMatcher,
    eval_df: pd.DataFrame,
    top_k_values: list[int] = [1, 3, 5],
    retrieve_top_k: int = 10,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for _, row in eval_df.iterrows():
        query_id = row["query_id"]
        resume_text = row["resume_text"]
        target_role_family = row["target_role_family"]
        difficulty = row["difficulty"]

        recommendations = matcher.recommend(
            resume_text,
            retrieve_top_k=retrieve_top_k,
            final_top_k=max(top_k_values),
        )
        predicted_roles = recommendations["role_family"].tolist()

        result_row = {
            "query_id": query_id,
            "difficulty": difficulty,
            "target_role_family": target_role_family,
            "top_1_role": predicted_roles[0] if len(predicted_roles) > 0 else None,
        }

        for k in top_k_values:
            result_row[f"hit@{k}"] = hit_at_k(predicted_roles, target_role_family, k)

        rows.append(result_row)

    results_df = pd.DataFrame(rows)

    metrics = {}
    for k in top_k_values:
        metrics[f"hit@{k}"] = results_df[f"hit@{k}"].mean()

    difficulty_metrics = (
        results_df.groupby("difficulty")[[f"hit@{k}" for k in top_k_values]]
        .mean()
        .reset_index()
    )

    return results_df, {"overall": metrics, "by_difficulty": difficulty_metrics}


def main():
    jobs_df = load_jobs("data/raw/jobs.csv")
    eval_df = load_eval_queries("data/raw/eval_queries_v2.csv")

    matcher = TwoStageJobMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        batch_size=32,
        retrieval_weight=0.55,
        skill_weight=0.20,
        domain_weight=0.15,
        title_weight=0.10,
    )
    matcher.fit(jobs_df)

    results_df, metrics = evaluate_matcher(
        matcher,
        eval_df,
        top_k_values=[1, 3, 5],
        retrieve_top_k=10,
    )

    print("\nPer-query results:\n")
    print(results_df.to_string(index=False))

    print("\nOverall metrics:\n")
    for metric_name, value in metrics["overall"].items():
        print(f"{metric_name}: {value:.3f}")

    print("\nMetrics by difficulty:\n")
    print(metrics["by_difficulty"].to_string(index=False))


if __name__ == "__main__":
    main()
