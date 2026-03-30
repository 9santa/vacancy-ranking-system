from pathlib import Path
import pandas as pd
import re


INPUT_PATH = "data/raw/jobs_v3_real.csv"
OUTPUT_PATH = "data/raw/jobs_v3_real_clean.csv"


WEAK_TITLE_PATTERNS = [
    r"\bjobs?\b",
    r"\bnow hiring\b",
    r"\burgent\b",
    r"\bhiring immediately\b",
]

AMBIGUOUS_MIXED_TITLE_PATTERNS = [
    r"data analyst.*data scientist",
    r"data scientist.*data analyst",
    r"data scientist.*ml engineer",
    r"ml engineer.*data scientist",
    r"data engineer.*data scientist",
    r"data scientist.*data engineer",
    r"business intelligence.*data analyst.*data scientist",
]

BI_ANALYST_KEYWORDS = [
    "business intelligence analyst",
    "business intelligence data analyst",
    "data analyst, business intelligence",
    "data analyst business intelligence",
    "senior data analyst, business intelligence",
    "senior business intelligence data analyst",
    "bi analyst",
    "lead bi analyst",
    "jr. business intelligence analyst",
    "power bi analyst",
    "reporting analyst",
    "data reporting analyst",
    "dashboard analyst",
    "kpi reporting",
    "reporting specialist",
]

BI_ENGINEERING_KEYWORDS = [
    "business intelligence data engineer",
    "data engineer, business intelligence",
    "data engineer business intelligence",
    "bi developer",
    "business intelligence developer",
    "reporting developer",
    "dashboard developer",
]

ANALYTICS_ENGINEERING_KEYWORDS = [
    "analytics engineer",
]

DATA_ENGINEERING_KEYWORDS = [
    "data engineer",
    "etl",
    "pipeline engineer",
    "data platform engineer",
    "big data engineer",
    "cloud data engineer",
]

ML_ENGINEERING_KEYWORDS = [
    "machine learning engineer",
    "ml engineer",
    "mlops",
    "ml ops",
    "model deployment",
    "model serving",
    "inference engineer",
]

DATA_SCIENCE_KEYWORDS = [
    "data scientist",
    "applied scientist",
    "research scientist",
    "predictive analytics",
    "machine learning scientist",
]

ANALYTICS_KEYWORDS = [
    "data analyst",
    "product analyst",
    "marketing analyst",
    "growth analyst",
    "analytics",
    "business analyst",
    "business insights analyst",
]

AMBIGUOUS_ML_TITLE_KEYWORDS = [
    "engineer",
    "machine learning",
    "phd position",
    "research",
]


def is_ambiguous_ml_title(title: str) -> bool:
    t = normalize_text(title)

    # good ML engineering titles
    if any(
        x in t
        for x in [
            "machine learning engineer",
            "ml engineer",
            "mlops",
            "ml ops",
            "ai/ml engineer",
            "senior ml engineer",
            "lead machine learning engineer",
            "staff machine learning engineer",
            "principal machine learning engineer",
            "junior machine learning engineer",
            "nlp engineer",
        ]
    ):
        return False

    # clearly noisy / too generic
    if t in {"engineer", "machine learning", "ai engineer"}:
        return True

    if "phd position" in t:
        return True

    if "research" in t and "engineer" not in t:
        return True

    if "etl engineer" in t:
        return True

    return False


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def parse_skills(skills_value: str) -> list[str]:
    if pd.isna(skills_value):
        return []

    skills = [s.strip().lower() for s in str(skills_value).split(";")]
    skills = [s for s in skills if s]
    return sorted(set(skills))


def is_weak_title(title: str) -> bool:
    t = normalize_text(title)
    return any(re.search(pattern, t) for pattern in WEAK_TITLE_PATTERNS)


def is_ambiguous_mixed_title(title: str) -> bool:
    t = normalize_text(title)

    if "/" in t:
        # Titles like "Data Analyst / Data Scientist" are too noisy
        # for the first clean training corpus.
        if (
            ("data analyst" in t and "data scientist" in t)
            or ("data scientist" in t and "ml engineer" in t)
            or ("data engineer" in t and "data scientist" in t)
        ):
            return True

    return any(re.search(pattern, t) for pattern in AMBIGUOUS_MIXED_TITLE_PATTERNS)


def map_family_and_subfamily(
    title: str, title_short: str
) -> tuple[str | None, str | None]:
    t = normalize_text(title)
    ts = normalize_text(title_short)

    # 1. Engineering-first rules for special subfamilies
    if contains_any(t, BI_ENGINEERING_KEYWORDS):
        return "data_engineering", "bi_engineering"

    if contains_any(t, ANALYTICS_ENGINEERING_KEYWORDS):
        return "data_engineering", "analytics_engineering"

    if contains_any(t, ML_ENGINEERING_KEYWORDS) or ts == "machine learning engineer":
        if "mlops" in t or "ml ops" in t:
            return "ml_engineering", "mlops"
        return "ml_engineering", "general_ml_engineering"

    if contains_any(t, DATA_ENGINEERING_KEYWORDS) or ts == "data engineer":
        return "data_engineering", "general_data_engineering"

    # 2. BI analyst/reporting layer
    if contains_any(t, BI_ANALYST_KEYWORDS):
        return "bi", "bi_reporting"

    # 3. Data science
    if contains_any(t, DATA_SCIENCE_KEYWORDS) or ts == "data scientist":
        return "data_science", "general_data_science"

    # 4. Analytics
    if contains_any(t, ANALYTICS_KEYWORDS) or ts in {
        "data analyst",
        "business analyst",
    }:
        if "product analyst" in t:
            return "analytics", "product_analytics"
        if "business analyst" in t:
            return "analytics", "business_analytics"
        return "analytics", "general_analytics"

    return None, None


def main():
    df = pd.read_csv(INPUT_PATH).copy()

    # Parse skills properly
    df["skill_list"] = df["skills"].apply(parse_skills)
    df["n_skills"] = df["skill_list"].apply(len)
    df["skills"] = df["skill_list"].apply(lambda xs: ";".join(xs))

    # Filter weak/noisy rows
    df = df[~df["title"].apply(is_weak_title)].copy()
    df = df[~df["title"].apply(is_ambiguous_mixed_title)].copy()

    # Remove rows with too little signal
    df = df[df["n_skills"] >= 2].copy()

    # Re-map family + subfamily using stricter rules
    mapped = df.apply(
        lambda row: map_family_and_subfamily(row["title"], row.get("title_short", "")),
        axis=1,
    )

    df["role_family_v2"] = [x[0] for x in mapped]
    df["role_subfamily"] = [x[1] for x in mapped]

    # Keep only rows we can map confidently
    df = df[df["role_family_v2"].notna()].copy()

    df["is_ambiguous"] = False
    df.loc[df["title"].apply(is_ambiguous_mixed_title), "is_ambiguous"] = True
    df = df[~df["is_ambiguous"]].copy()

    # Remove noisy ML-engineering titles
    ml_noise_mask = (df["role_family_v2"] == "ml_engineering") & df["title"].apply(
        is_ambiguous_ml_title
    )
    df = df[~ml_noise_mask].copy()

    # Replace old family with cleaned one
    df["role_family"] = df["role_family_v2"]
    df = df.drop(columns=["role_family_v2"])

    # Rebuild description to keep it aligned with cleaned skills
    def rebuild_description(row):
        return (
            f"Title short: {row.get('title_short', '')}. "
            f"Schedule: {row.get('schedule_type', '')}. "
            f"Remote: {row.get('work_from_home', '')}. "
            f"Country: {row.get('country', '')}. "
            f"Skills: {', '.join(row['skill_list'])}."
        )

    df["description"] = df.apply(rebuild_description, axis=1)

    # Drop duplicates again after cleanup
    df = df.drop_duplicates(subset=["title", "company", "location", "skills"]).copy()

    # Reassign job_id
    df = df.reset_index(drop=True)
    df["job_id"] = range(1, len(df) + 1)

    out_cols = [
        "job_id",
        "title",
        "company",
        "location",
        "description",
        "skills",
        "role_family",
        "role_subfamily",
        "raw_title",
        "title_short",
        "country",
        "schedule_type",
        "work_from_home",
        "source",
        "n_skills",
    ]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print("\n=== CLEAN SHAPE ===")
    print(df.shape)

    print("\n=== ROLE FAMILY COUNTS ===")
    print(df["role_family"].value_counts().to_string())

    print("\n=== ROLE SUBFAMILY COUNTS ===")
    print(df["role_subfamily"].value_counts().to_string())

    print("\n=== SAMPLE ROWS ===")
    print(
        df[["job_id", "title", "role_family", "role_subfamily", "skills"]]
        .sample(min(20, len(df)), random_state=42)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
