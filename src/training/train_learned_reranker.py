from pathlib import Path
import pandas as pd

from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher

def load_queries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["query_id", "resume_text", "target_role_family"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in query file: {missing_columns}")

    return df


def build_pair_dataset(
    matcher: TwoStageLearnedMatcher,
    queries_df: pd.DataFrame,
    retrieve_top_k: int = 30,
    max_positives_per_query: int = 2,
    max_negatives_per_query: int = 6,
) -> pd.DataFrame:
    """
    Builds a more balanced pair dataset for learned reranking.

    Strategy:
    - retrieve a wider shortlist
    - keep only a few top positives
    - keep several top hard negatives

    This makes the reranker learn to distinguish
    correct jobs from close-but-wrong alternatives.
    """
    rows = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        resume_text = row["resume_text"]
        target_role_family = row["target_role_family"]

        # Wider retrieval pool for mining hard negatives
        candidates = matcher.retriever.recommend(resume_text, top_k=retrieve_top_k)

        positives = (
            candidates[candidates["role_family"] == target_role_family]
            .sort_values("score", ascending=False)
            .head(max_positives_per_query)
        )

        hard_negatives = (
            candidates[candidates["role_family"] != target_role_family]
            .sort_values("score", ascending=False)
            .head(max_negatives_per_query)
        )

        selected_candidates = pd.concat([positives, hard_negatives], ignore_index=True)

        # If for some reason nothing selected, skip query
        if selected_candidates.empty:
            continue

        feature_df = matcher.reranker.build_feature_frame(resume_text, selected_candidates)

        feature_df["query_id"] = query_id
        feature_df["target_role_family"] = target_role_family
        feature_df["label"] = (
            feature_df["role_family"] == target_role_family
        ).astype(int)

        rows.append(feature_df)

    # pair_df = pd.concat(rows, ignore_index=True)
    return pd.concat(rows, ignore_index=True)


def hit_at_k(recommended_role_families: list[str], target_role_family: str, k: int) -> int:
    return int(target_role_family in recommended_role_families[:k])


def evaluate_matcher(
    matcher: TwoStageLearnedMatcher,
    eval_df: pd.DataFrame,
    retrieve_top_k: int = 10,
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
            "difficulty": row.get("difficulty", "unknown"),
            "target_role_family": row["target_role_family"],
            "top_1_role": predicted_roles[0] if predicted_roles else None,
        }

        for k in top_k_values:
            result_row[f"hit@{k}"] = hit_at_k(predicted_roles, row["target_role_family"], k)

        rows.append(result_row)

    results_df = pd.DataFrame(rows)
    metrics = {f"hit@{k}": results_df[f"hit@{k}"].mean() for k in top_k_values}
    return results_df, metrics


def main():
    jobs_df = load_jobs("data/raw/jobs_v2.csv")
    train_df = load_queries("data/raw/train_queries_v2.csv")
    val_df = load_queries("data/raw/val_queries_v2.csv")

    matcher = TwoStageLearnedMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512,
    )
    matcher.fit_jobs(jobs_df)

    train_pair_df = build_pair_dataset(matcher, train_df, retrieve_top_k=30, max_positives_per_query=2, max_negatives_per_query=6)

    print("\nTraining pair dataset info:")
    print(f"Rows: {len(train_pair_df)}")
    print(f"Positive labels: {train_pair_df['label'].sum()}")
    print(f"Negative labels: {(train_pair_df['label'] == 0).sum()}")
    print("\nLabel distribution:")
    print(train_pair_df["label"].value_counts(normalize=True).to_string())

    matcher.fit_reranker(train_pair_df)

    results_df, metrics = evaluate_matcher(matcher, val_df, retrieve_top_k=10, top_k_values=[1, 3, 5])

    print("\nValidation results:\n")
    print(results_df.to_string(index=False))

    print("\nValidation metrics:\n")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

    artifact_path = Path("artifacts/learned_reranker.joblib")
    matcher.reranker.save(artifact_path)
    print(f"\nSaved trained reranker to: {artifact_path}")


if __name__ == "__main__":
    main()
