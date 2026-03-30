import pandas as pd


JOBS_PATH = "data/raw/jobs_v3_real.csv"


def show_examples(df: pd.DataFrame, mask, title: str, n: int = 20):
    subset = df[mask].copy()
    print(f"\n=== {title} ({len(subset)}) ===")
    if subset.empty:
        print("No rows found.")
        return
    cols = ["job_id", "title", "role_family", "company", "location", "skills"]
    print(subset[cols].head(n).to_string(index=False))


def main():
    df = pd.read_csv(JOBS_PATH)

    print("\n=== DATASET INFO ===")
    print(df.shape)

    print("\n=== ROLE FAMILY COUNTS ===")
    print(df["role_family"].value_counts().to_string())

    # suspicious title-family combinations
    show_examples(
        df,
        df["title"].str.contains(
            "business intelligence|bi analyst|reporting analyst|dashboard analyst",
            case=False,
            na=False,
        )
        & (df["role_family"] != "bi"),
        "BI-like titles mapped outside BI",
    )

    show_examples(
        df,
        df["title"].str.contains(
            "data scientist|applied scientist|machine learning scientist|research scientist",
            case=False,
            na=False,
        )
        & (df["role_family"] != "data_science"),
        "Data-science-like titles mapped outside data_science",
    )

    show_examples(
        df,
        df["title"].str.contains(
            "machine learning engineer|ml engineer|mlops|model serving|inference engineer",
            case=False,
            na=False,
        )
        & (df["role_family"] != "ml_engineering"),
        "ML-engineering-like titles mapped outside ml_engineering",
    )

    show_examples(
        df,
        df["title"].str.contains(
            "data engineer|analytics engineer|etl|pipeline engineer|data platform",
            case=False,
            na=False,
        )
        & (df["role_family"] != "data_engineering"),
        "Data-engineering-like titles mapped outside data_engineering",
    )

    show_examples(
        df,
        df["title"].str.contains(
            "data analyst|product analyst|marketing analyst|growth analyst|analytics",
            case=False,
            na=False,
        )
        & (df["role_family"] != "analytics"),
        "Analytics-like titles mapped outside analytics",
    )

    # empty / weak rows
    show_examples(
        df,
        df["skills"].fillna("").str.len() < 5,
        "Rows with almost empty skills",
    )

    # duplicates by core fields
    dup_mask = df.duplicated(
        subset=["title", "company", "location", "skills"], keep=False
    )
    show_examples(
        df,
        dup_mask,
        "Potential duplicates",
    )

    # title_short distribution if present
    if "title_short" in df.columns:
        print("\n=== TITLE_SHORT by ROLE_FAMILY ===")
        print(
            df.groupby(["role_family", "title_short"])
            .size()
            .sort_values(ascending=False)
            .head(50)
            .to_string()
        )


if __name__ == "__main__":
    main()
