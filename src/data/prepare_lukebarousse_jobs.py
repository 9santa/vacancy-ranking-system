from pathlib import Path
import ast
import pandas as pd
from datasets import load_dataset

OUT_PATH = "data/raw/jobs_v3_real.csv"

TITLE_SHORT_TO_FAMILY = {
    "Data Scientist": "data_science",
    "Data Analyst": "analytics",
    "Business Analyst": "analytics",
    "Data Engineer": "data_engineering",
    "Machine Learning Engineer": "ml_engineering",
}


def map_role_family(title_short: str, title: str) -> str | None:
    ts = str(title_short).strip().lower()
    t = str(title).strip().lower()

    # direcct normalized title_short
    if ts == "data scientist":
        return "data_science"
    if ts in {"data analyst", "business analyst"}:
        return "analytics"
    if ts == "data engineer":
        # TODO: later
        pass
    if ts == "machine learning engineer":
        return "ml_engineering"

    # strong BI keywords
    if any(
        x in t
        for x in [
            "business intelligence",
            "bi analyst",
            "reporting analyst",
            "dashboard analyst",
            "kpi reporting",
            "power bi analyst",
        ]
    ):
        return "bi"

    # strong ML engineering keywords
    if any(
        x in t
        for x in [
            "ml engineer",
            "machine learning engineer",
            "mlops",
            "model deployment",
            "model serving",
            "inference engineer",
        ]
    ):
        return "ml_engineering"

    # strong data engineering keywords
    if any(
        x in t
        for x in [
            "data engineer",
            "analytics engineer",
            "etl",
            "pipeline engineer",
            "data platform engineer",
        ]
    ):
        return "data_engineering"

    # strong data science keywords
    if any(
        x in t
        for x in [
            "data scientist",
            "applied scientist",
            "research scientist",
            "predictive analytics",
            "machine learning scientist",
        ]
    ):
        return "data_science"

    # strong analytics keywords
    if any(
        x in t
        for x in [
            "data analyst",
            "product analyst",
            "marketing analyst",
            "growth analyst",
            "analytics",
            "business analyst",
        ]
    ):
        return "analytics"

    return None


def parse_job_skills(value: str) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return sorted({str(x).strip().lower() for x in parsed})
    except Exception:
        pass
    return []


def parse_job_type_skills(value: str) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            out = []
            for _, vals in parsed.items():
                if isinstance(vals, list):
                    out.extend([str(x).strip().lower() for x in vals])
            return sorted(set(out))
    except Exception:
        pass
    return []


def merge_skills(row) -> list[str]:
    skills = parse_job_skills(row["job_skills"])
    if not skills:
        skills = parse_job_type_skills(row["job_type_skills"])
    return skills


# raw data doesn't have a description field
# we'll create compact synthetic description
def build_description(row, skill_list: list[str]) -> str:
    return (
        f"Title short: {row['job_title_short']}. "
        f"Schedule: {row['job_schedule_type']}. "
        f"Remote: {row['job_work_from_home']}. "
        f"Country: {row['job_country']}. "
        f"Skills: {', '.join(skill_list)}."
    )


def main():
    ds = load_dataset("lukebarousse/data_jobs", split="train")
    df = ds.to_pandas()

    keep_cols = [
        "job_title_short",
        "job_title",
        "job_location",
        "job_via",
        "job_schedule_type",
        "job_work_from_home",
        "job_country",
        "company_name",
        "job_skills",
        "job_type_skills",
    ]
    df = df[keep_cols].copy()

    df["role_family"] = df.apply(
        lambda row: map_role_family(row["job_title_short"], row["job_title"]),
        axis=1,
    )
    df = df[df["role_family"].notna()].copy()

    df["skill_list"] = df.apply(merge_skills, axis=1)
    df["skills"] = df["skill_list"].apply(lambda xs: ";".join(xs[:20]))
    df["description"] = df.apply(
        lambda row: build_description(row, row["skill_list"][:20]), axis=1
    )

    df["title"] = df["job_title"]
    df["company"] = df["company_name"]
    df["location"] = df["job_location"]
    df["raw_title"] = df["job_title"]
    df["title_short"] = df["job_title_short"]
    df["country"] = df["job_country"]
    df["schedule_type"] = df["job_schedule_type"]
    df["work_from_home"] = df["job_work_from_home"]
    df["source"] = "lukebarousse/data_jobs"

    # basic cleanup
    df = df[df["skills"].str.len() > 0]
    df = df.drop_duplicates(subset=["title", "company", "location", "skills"]).copy()

    # optional balancing cap for first pass
    caps = {
        "data_science": 500,
        "analytics": 700,
        "bi": 300,
        "data_engineering": 700,
        "ml_engineering": 300,
    }

    parts = []
    for family, cap in caps.items():
        part = df[df["role_family"] == family].head(cap)
        parts.append(part)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.reset_index(drop=True)
    out_df.insert(0, "job_id", range(1, len(out_df) + 1))

    out_cols = [
        "job_id",
        "title",
        "company",
        "location",
        "description",
        "skills",
        "role_family",
        "raw_title",
        "title_short",
        "country",
        "schedule_type",
        "work_from_home",
        "source",
    ]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_df[out_cols].to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("\nRole family counts:\n")
    print(out_df["role_family"].value_counts().to_string())


if __name__ == "__main__":
    main()
