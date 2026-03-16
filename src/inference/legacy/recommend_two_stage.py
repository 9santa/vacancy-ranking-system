from src.data.load_data import load_jobs
from src.models.two_stage_matcher import TwoStageJobMatcher

def main():
    jobs_df = load_jobs("data/raw/jobs.csv")

    matcher = TwoStageJobMatcher(
        embedding_model_name="all-MiniLM-L6-v2",
        batch_size=32,
        retrieval_weight=0.55,
        skill_weight=0.20,
        domain_weight=0.15,
        title_weight=0.10,
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
        final_top_k=5
    )

    print("\nTop recommendations:\n")
    print(
        recommendations[
            [
                "job_id",
                "title",
                "rerank_score",
                "retrieval_score",
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
