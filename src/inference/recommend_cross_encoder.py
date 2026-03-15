from src.data.load_data import load_jobs
from src.models.two_stage_cross_encoder_matcher import TwoStageCrossEncoderMatcher


def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = TwoStageCrossEncoderMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size=32,
        reranker_batch_size=16,
        reranker_max_length=512,
    )
    matcher.fit(jobs_df)

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
                "reranker_score",
                "retrieval_score",
                "matched_skills",
                "role_family",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
