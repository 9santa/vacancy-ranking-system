from src.data.load_data import load_jobs
from src.models.two_stage_learned_matcher import TwoStageLearnedMatcher


JOBS_PATH = "data/raw/jobs_v3_real_clean.csv"
MODEL_ARTIFACT_PATH = "artifacts/learned_reranker_no_domain.joblib"


TEST_QUERIES = {
    "data_science": """
    Worked on machine learning, feature engineering, forecasting, and model evaluation in Python.
    Built predictive models on structured datasets and communicated findings to stakeholders.
    """,
    "analytics": """
    Worked on A/B testing, retention analysis, segmentation, SQL dashboards, and product decision support.
    Focused on business insights, stakeholder questions, and ad hoc analysis.
    """,
    "bi": """
    Built Power BI dashboards, supported recurring KPI reporting, maintained executive dashboards,
    and worked on stakeholder reporting workflows using SQL and Excel.
    """,
    "data_engineering": """
    Built ETL pipelines with Airflow and Spark, maintained warehouse tables, worked with dbt and SQL,
    and supported reliable data delivery for downstream analytics teams.
    """,
    "ml_engineering": """
    Worked on model deployment, inference services, training pipelines, Docker, Kubernetes,
    and monitoring machine learning models in production.
    """,
}


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

    for query_name, resume_text in TEST_QUERIES.items():
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query_name}")
        print(f"{'=' * 80}")

        recommendations = matcher.recommend(
            resume_text,
            retrieve_top_k=30,
            final_top_k=5,
        )

        print(
            recommendations[
                [
                    "job_id",
                    "title",
                    "role_family",
                    "role_subfamily",
                    "learned_score",
                    "retrieval_score",
                    "cross_encoder_score",
                    "skill_overlap_bonus",
                    "title_alignment_bonus",
                    "matched_skills",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
