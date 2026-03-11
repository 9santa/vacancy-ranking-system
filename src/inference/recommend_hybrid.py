from src.data.load_data import load_jobs
from src.models.hybrid_tfidf_matcher import HybridTfidfJobMatcher

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = HybridTfidfJobMatcher(
        title_weight=0.45,
        skills_weight=0.35,
        description_weight=0.20,
        base_score_weight=0.70,
        skill_bonus_weight=0.15,
        domain_bonus_weight=0.10,
        family_bonus_weight=0.05,
        max_features=5000,
        ngram_range=(1, 2),
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
                "structured_score",
                "bonus_score",
                "skill_overlap_bonus",
                "domain_phrase_bonus",
                "family_alignment_bonus",
                "matched_skills",
                "matched_domain_terms",
                "inferred_resume_family",
                "inferred_job_family",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
