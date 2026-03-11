from src.data.load_data import load_jobs
from src.models.structured_tfidf_matcher import StructuredTfidfJobMatcher

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = StructuredTfidfJobMatcher(
        title_weight=0.45,
        skills_weight=0.35,
        description_weight=0.20,
        max_features=5000,
        ngram_range=(1, 2)
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
                "title_score",
                "skills_score",
                "description_score",
                "matched_skills",
                "title_overlap_terms",
                "skills_overlap_terms",
                "description_overlap_terms",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()

