import pandas as pd

from src.data.load_data import load_jobs, build_job_text
from src.models.tfidf_matcher import TfidfJobMatcher

# Load eval data into df, return DataFrame
def load_eval_queries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["query_id", "resume_text", "target_role_family"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in eval file: {missing_columns}")

    return df

# Returns 1, if within top-k there is atleast one vacancy with correct class. Else 0.
def hit_at_k(recommended_role_families: list[str], target_role_family: str, k: int) -> int:
    top_k_roles = recommended_role_families[:k]
    return int(target_role_family in top_k_roles)

# Run matcher on eval-queries and calc simple metrics (hit@k)
def evaluate_matcher(matcher: TfidfJobMatcher, eval_df: pd.DataFrame, top_k_values: list[int] = [3, 5]) -> tuple[pd.DataFrame, dict]:
    rows = []

    for _, row in eval_df.iterrows():
        query_id = row["query_id"]
        resume_text = row["resume_text"]
        target_role_family = row["target_role_family"]

        recommendations = matcher.recommend(resume_text, top_k=max(top_k_values))
        predicted_roles = recommendations["role_family"].tolist()

        result_row = {
            "query_id": query_id,
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

    return (results_df, metrics)

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")
    jobs_df = build_job_text(jobs_df)

    eval_df = load_eval_queries("data/raw/eval_queries.csv")

    matcher = TfidfJobMatcher(max_features=5000, ngram_range=(1, 2))
    matcher.fit(jobs_df)

    results_df, metrics = evaluate_matcher(matcher, eval_df, top_k_values=[1, 3, 5])

    print("\nPer-query results:\n")
    print(results_df.to_string(index=False))

    print("\nAggregate metrics:\n")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

if __name__ == "__main__":
    main()
