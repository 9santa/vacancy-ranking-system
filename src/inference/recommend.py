from src.data.load_data import load_jobs, build_job_text
from src.models.tfidf_matcher import TfidfJobMatcher

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")
    jobs_df = build_job_text(jobs_df)

    matcher = TfidfJobMatcher(max_features=5000, ngram_range=(1, 2))
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
                "overlap_terms"
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()
