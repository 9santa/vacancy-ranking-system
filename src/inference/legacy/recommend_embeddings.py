from src.data.load_data import load_jobs
from src.models.embedding_matcher import EmbeddingJobMatcher

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = EmbeddingJobMatcher(
        model_name="all-MiniLM-L6-v2",
        batch_size=32
    )
    matcher.fit(jobs_df)

    sample_resume = """
    Data Science student with experience in Python, SQL, pandas, data visualization,
    machine learning basics, A/B testing, and dashboard development.
    Worked on internship-style projects involving data cleaning, analytics, and statistics.
    """

    recommendations = matcher.recommend(sample_resume, top_k=5)

    print("\nTop recommendations:\n")
    print(
        recommendations[
            [
                "job_id",
                "title",
                "score",
                "matched_skills",
                "role_family",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
