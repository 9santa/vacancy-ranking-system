from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["job_id", "title", "company", "location", "description", "skills"]

def load_jobs(csv_path: str | Path) -> pd.DataFrame:
    # Load CSV with vanancies, ensure required columns are present
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df

# Build one full text from columns
def build_job_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = df[col].fillna("").astype(str)

    df["job_text"] = (
        (df["title"] + " ") * 3
        + (df["skills"] + " ") * 2
        + df["description"]
    )

    return df

