from pathlib import Path
import pandas as pd


TRAIN_ROWS = [
    # data_science
    {
        "resume_text": "Computer science student with Python, pandas, scikit-learn, machine learning, regression, classification, and model evaluation experience.",
        "target_role_family": "data_science",
        "difficulty": "easy",
    },
    {
        "resume_text": "Built forecasting models in Python, used statistics and feature engineering, and evaluated predictive performance on tabular data.",
        "target_role_family": "data_science",
        "difficulty": "easy",
    },
    {
        "resume_text": "Worked on NLP tasks, embeddings, text preprocessing, and exploratory analysis with pandas and Python.",
        "target_role_family": "data_science",
        "difficulty": "medium",
    },
    {
        "resume_text": "Used machine learning, experiment analysis, and statistical modeling to support predictive analytics projects.",
        "target_role_family": "data_science",
        "difficulty": "medium",
    },
    {
        "resume_text": "Built churn and forecasting models, but also created dashboards and explained model outputs to business teams.",
        "target_role_family": "data_science",
        "difficulty": "hard",
    },
    {
        "resume_text": "Worked with feature engineering, gradient boosting, SQL, and executive presentations for model-driven decisions.",
        "target_role_family": "data_science",
        "difficulty": "hard",
    },

    # analytics
    {
        "resume_text": "Analyzed product metrics, retention, funnel performance, and A/B test results using SQL and Python.",
        "target_role_family": "analytics",
        "difficulty": "easy",
    },
    {
        "resume_text": "Worked on campaign performance analysis, segmentation, and marketing reporting with Tableau and SQL.",
        "target_role_family": "analytics",
        "difficulty": "easy",
    },
    {
        "resume_text": "Built dashboards for growth teams, tracked KPIs, and performed ad hoc analysis for product decisions.",
        "target_role_family": "analytics",
        "difficulty": "medium",
    },
    {
        "resume_text": "Used SQL, Python, and experimentation results to analyze user behavior, retention, and conversion funnels.",
        "target_role_family": "analytics",
        "difficulty": "medium",
    },
    {
        "resume_text": "Owned weekly KPI dashboards, segmentation analysis, and stakeholder reporting for growth and retention teams.",
        "target_role_family": "analytics",
        "difficulty": "hard",
    },
    {
        "resume_text": "Worked with dashboards, A/B tests, user behavior analysis, and business recommendations for product managers.",
        "target_role_family": "analytics",
        "difficulty": "hard",
    },

    # bi
    {
        "resume_text": "Built Power BI dashboards, maintained recurring reports, monitored KPIs, and supported business intelligence workflows.",
        "target_role_family": "bi",
        "difficulty": "easy",
    },
    {
        "resume_text": "Prepared recurring executive reports, validated source data, and maintained business dashboards in Power BI.",
        "target_role_family": "bi",
        "difficulty": "easy",
    },
    {
        "resume_text": "Worked on reporting pipelines, dashboard maintenance, KPI tracking, and operational business reporting.",
        "target_role_family": "bi",
        "difficulty": "medium",
    },
    {
        "resume_text": "Maintained reporting workflows, created dashboards for finance teams, and supported recurring KPI reviews.",
        "target_role_family": "bi",
        "difficulty": "medium",
    },
    {
        "resume_text": "Supported business intelligence reporting, dashboard refresh cycles, KPI definitions, and recurring stakeholder reports.",
        "target_role_family": "bi",
        "difficulty": "hard",
    },
    {
        "resume_text": "Worked with Power BI, SQL dashboards, recurring business reports, and executive KPI monitoring for operations teams.",
        "target_role_family": "bi",
        "difficulty": "hard",
    },
]

VAL_ROWS = [
    # data_science
    {
        "resume_text": "Applied machine learning, feature engineering, and statistics to build predictive models in Python.",
        "target_role_family": "data_science",
        "difficulty": "easy",
    },
    {
        "resume_text": "Worked on forecasting and classification problems, but also created dashboards for communicating model results.",
        "target_role_family": "data_science",
        "difficulty": "hard",
    },
    {
        "resume_text": "Used Python, pandas, model evaluation, and experiment analysis for data science projects.",
        "target_role_family": "data_science",
        "difficulty": "medium",
    },

    # analytics
    {
        "resume_text": "Analyzed user funnels, retention, and A/B tests using SQL, dashboards, and product metrics.",
        "target_role_family": "analytics",
        "difficulty": "easy",
    },
    {
        "resume_text": "Built KPI dashboards and ran ad hoc analysis for product and growth stakeholders.",
        "target_role_family": "analytics",
        "difficulty": "medium",
    },
    {
        "resume_text": "Worked on segmentation, experimentation, dashboards, and recurring stakeholder insights for business teams.",
        "target_role_family": "analytics",
        "difficulty": "hard",
    },

    # bi
    {
        "resume_text": "Created Power BI dashboards, monitored KPI quality, and supported recurring business reporting.",
        "target_role_family": "bi",
        "difficulty": "easy",
    },
    {
        "resume_text": "Maintained reporting pipelines, executive dashboards, and recurring KPI reporting for leadership teams.",
        "target_role_family": "bi",
        "difficulty": "medium",
    },
    {
        "resume_text": "Worked on SQL reporting views, dashboard refreshes, and business intelligence support for recurring reporting.",
        "target_role_family": "bi",
        "difficulty": "hard",
    },
]


def build_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df.insert(0, "query_id", range(1, len(df) + 1))
    return df


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_df(TRAIN_ROWS)
    val_df = build_df(VAL_ROWS)

    train_df.to_csv(out_dir / "train_queries_v1.csv", index=False)
    val_df.to_csv(out_dir / "val_queries_v1.csv", index=False)

    print("Saved:")
    print(out_dir / "train_queries_v1.csv")
    print(out_dir / "val_queries_v1.csv")


if __name__ == "__main__":
    main()
