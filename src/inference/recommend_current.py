from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher


JOBS_PATH = "data/raw/jobs_v2.csv"
MODEL_ARTIFACT_PATH = "artifacts/learned_reranker_no_domain.joblib"


def main():
    jobs_df = load_jobs(JOBS_PATH)

    matcher = TwoStageLearnedMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512,
    )
    matcher.fit_jobs(jobs_df)
    matcher.reranker.load(MODEL_ARTIFACT_PATH)

    sample_resume = """
    Worked on SQL analysis, A/B testing, retention, segmentation, and dashboard-based
    reporting for product teams. Built recommendations for stakeholders and supported
    experiment analysis with Python and business metrics.
    """

    recommendations = matcher.recommend(
        sample_resume,
        retrieve_top_k=30,
        final_top_k=5,
    )

    print("\nCurrent model recommendations:\n")
    print(
        recommendations[
            [
                "job_id",
                "title",
                "learned_score",
                "retrieval_score",
                "cross_encoder_score",
                "skill_overlap_bonus",
                "title_alignment_bonus",
                "matched_skills",
                "role_family",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
