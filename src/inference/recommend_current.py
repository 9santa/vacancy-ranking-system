from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher


def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = TwoStageLearnedMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        cross_encoder_batch_size=16,
        cross_encoder_max_length=512,
    )
    matcher.fit_jobs(jobs_df)
    matcher.reranker.load("artifacts/learned_reranker.joblib")

    sample_resume = """
    Data Science student with experience in Python, SQL, pandas, data visualization,
    machine learning basics, A/B testing, and dashboard development.
    Worked on internship-style projects involving data cleaning, analytics, and statistics.
    """

    recommendations = matcher.recommend(
        sample_resume,
        retrieve_top_k=10,
        final_top_k=5,
    )

    print("\nTop recommendations:\n")
    print(
        recommendations[
            [
                "job_id",
                "title",
                "learned_score",
                "retrieval_score",
                "cross_encoder_score",
                "skill_overlap_bonus",
                "domain_phrase_bonus",
                "title_alignment_bonus",
                "matched_skills",
                "matched_domain_terms",
                "role_family",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
