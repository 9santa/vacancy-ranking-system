import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import norm

from src.preprocessing.text_cleaning import clean_text

class EmbeddingJobMatcher:
    """
    Semantic retrieval baseline.
    Idea:
    - using pre-trained sentence embeddings
    - encode vacancies and resume in dense vectors
    - compare via cosine similarity
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

        self.jobs_df = None
        self.job_embeddings = None

    def _build_job_text(self, df: pd.DataFrame) -> pd.Series:
        """
        Build single vacancy text.
        """
        df = df.copy()

        for col in ["title", "skills", "description"]:
            df[col] = df[col].fillna("").astype(str)

        job_text = (
            df["title"] + ". "
            + df["skills"] + ". "
            + df["description"]
        )

        return job_text.apply(clean_text)

    def _extract_job_skills(self, skills_text: str) -> set[str]:
        """
        Transform 'python;sql;excel;tableau'
        into set of tokens.
        """
        if not isinstance(skills_text, str):
            return set()

        skills = [skill.strip().lower() for skill in skills_text.split(";")]
        return set(skills)

    # NOTE: This is not 'Named-entity recognition', and not 'real' skill extractor
    def _get_matched_skills(self, resume_text: str, skills_text: str) -> list[str]:
        # Find intersection between resume tokens and vacancy's skills
        cleaned_resume = f" {clean_text(resume_text)} "
        job_skills = self._extract_job_skills(skills_text)

        matched = []
        for skill in job_skills:
            if f" {skill} " in cleaned_resume:
                matched.append(skill)

        return sorted(set(matched))

    def fit(self, jobs_df: pd.DataFrame) -> None:
        """
        Encode all vacancies one time.
        """
        required_columns = ["title", "skills", "description"]
        missing_columns = [col for col in required_columns if col not in jobs_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for embedding matcher: {missing_columns}")

        self.jobs_df = jobs_df.copy()
        job_texts = self._build_job_text(self.jobs_df).tolist()

        self.job_embeddings = self.model.encode(
            job_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True, # normalize so that cosine similarity actually works
            show_progress_bar=False
        )

    def recommend(self, resume_text: str, top_k: int = 10) -> pd.DataFrame:
        """
        Returns top-k semantic similarity relevant vacancies
        """
        if self.jobs_df is None or self.job_embeddings is None:
            raise ValueError("Mode is not fitted. Call fit() first.")

        cleaned_resume = clean_text(resume_text)

        resume_embedding = self.model.encode(
            [cleaned_resume],
            convert_to_numpy=True,
            normalize_embeddings=True, # same as before
            show_progress_bar=False
        )

        similarities = cosine_similarity(resume_embedding, self.job_embeddings).flatten()

        results = self.jobs_df.copy()
        results["score"] = similarities

        matched_skills_list = []
        for _, row in results.iterrows():
            matched_skills = self._get_matched_skills(resume_text, row["skills"])
            matched_skills_list.append(", ".join(matched_skills))

        results["matched_skills"] = matched_skills_list
        results = results.sort_values("score", ascending=False).head(top_k)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "score",
            "matched_skills",
            "description",
            "skills",
        ]

        if "role_family" in results.columns:
            return_columns.append("role_family")

        return results[return_columns]
