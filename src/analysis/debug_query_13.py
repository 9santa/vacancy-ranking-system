import numpy as np
import pandas as pd

from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher


JOBS_PATH = "data/raw/jobs_v2.csv"
TEST_QUERIES_PATH = "data/raw/test_queries_v3.csv"
MODEL_ARTIFACT_PATH = "artifacts/learned_reranker_no_domain.joblib"

TARGET_QUERY_ID = 13


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def main():
    jobs_df = load_jobs(JOBS_PATH)
    test_df = pd.read_csv(TEST_QUERIES_PATH)

    query_row = test_df[test_df["query_id"] == TARGET_QUERY_ID]
    if query_row.empty:
        raise ValueError(
            f"Query with id={TARGET_QUERY_ID} not found in {TEST_QUERIES_PATH}"
        )

    query_row = query_row.iloc[0]
    resume_text = query_row["resume_text"]
    target_role_family = query_row["target_role_family"]
    difficulty = query_row.get("difficulty", "unknown")

    print("\n=== QUERY INFO ===")
    print(f"query_id: {TARGET_QUERY_ID}")
    print(f"target_role_family: {target_role_family}")
    print(f"difficulty: {difficulty}")
    print("\nresume_text:\n")
    print(resume_text)

    matcher = TwoStageLearnedMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512,
    )
    matcher.fit_jobs(jobs_df)
    matcher.reranker.load(MODEL_ARTIFACT_PATH)

    # Step 1: retrieval shortlist
    candidates = matcher.retriever.recommend(resume_text, top_k=30)

    print("\n=== RETRIEVER TOP-10 ===\n")
    print(
        candidates[
            [
                "job_id",
                "title",
                "score",
                "matched_skills",
                "role_family",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # Step 2: build feature frame before final reranking
    feature_df = matcher.reranker.build_feature_frame(resume_text, candidates).copy()

    # Step 3: compute learned scores manually and inspect contributions
    pipeline = matcher.reranker.model
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    feature_names = matcher.reranker.FEATURE_COLUMNS

    X = feature_df[feature_names]
    X_scaled = scaler.transform(X)

    coef = clf.coef_[0]
    intercept = clf.intercept_[0]

    contribution_cols = []
    for i, feat in enumerate(feature_names):
        col_name = f"{feat}_contrib"
        feature_df[col_name] = X_scaled[:, i] * coef[i]
        contribution_cols.append(col_name)

    feature_df["logit"] = feature_df[contribution_cols].sum(axis=1) + intercept
    feature_df["learned_score_manual"] = feature_df["logit"].apply(sigmoid)

    # Also compare to model prediction for sanity
    feature_df["learned_score"] = matcher.reranker.predict_scores(feature_df)

    feature_df = feature_df.sort_values("learned_score", ascending=False)

    print("\n=== FINAL RERANKED TOP-10 ===\n")
    print(
        feature_df[
            [
                "job_id",
                "title",
                "role_family",
                "learned_score",
                "retrieval_score",
                "cross_encoder_score",
                "skill_overlap_bonus",
                "title_alignment_bonus",
                "matched_skills",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== FEATURE CONTRIBUTIONS FOR TOP-5 ===\n")
    print(
        feature_df[
            [
                "job_id",
                "title",
                "role_family",
                "learned_score",
                "retrieval_score_norm",
                "cross_encoder_score_norm",
                "skill_overlap_bonus",
                "title_alignment_bonus",
                "retrieval_score_norm_contrib",
                "cross_encoder_score_norm_contrib",
                "skill_overlap_bonus_contrib",
                "title_alignment_bonus_contrib",
                "logit",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # Compare best analytics candidate vs best bi candidate
    best_analytics = feature_df[feature_df["role_family"] == "analytics"].head(1)
    best_bi = feature_df[feature_df["role_family"] == "bi"].head(1)

    print("\n=== BEST ANALYTICS CANDIDATE ===\n")
    if not best_analytics.empty:
        print(
            best_analytics[
                [
                    "job_id",
                    "title",
                    "role_family",
                    "learned_score",
                    "retrieval_score",
                    "cross_encoder_score",
                    "skill_overlap_bonus",
                    "title_alignment_bonus",
                    "matched_skills",
                    "retrieval_score_norm_contrib",
                    "cross_encoder_score_norm_contrib",
                    "skill_overlap_bonus_contrib",
                    "title_alignment_bonus_contrib",
                    "logit",
                ]
            ].to_string(index=False)
        )
    else:
        print("No analytics candidate found.")

    print("\n=== BEST BI CANDIDATE ===\n")
    if not best_bi.empty:
        print(
            best_bi[
                [
                    "job_id",
                    "title",
                    "role_family",
                    "learned_score",
                    "retrieval_score",
                    "cross_encoder_score",
                    "skill_overlap_bonus",
                    "title_alignment_bonus",
                    "matched_skills",
                    "retrieval_score_norm_contrib",
                    "cross_encoder_score_norm_contrib",
                    "skill_overlap_bonus_contrib",
                    "title_alignment_bonus_contrib",
                    "logit",
                ]
            ].to_string(index=False)
        )
    else:
        print("No bi candidate found.")


if __name__ == "__main__":
    main()
