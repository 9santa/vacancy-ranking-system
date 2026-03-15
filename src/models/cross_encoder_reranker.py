import pandas as pd
from sentence_transformers import CrossEncoder

from src.preprocessing.text_cleaning import clean_text


class CrossEncoderReranker:
    """
    Modern neural reranker over top-k candidates (pair-scorer).

    Idea:
    - we take shortlist from embedding retriever
    - for each pair (resume, vacancy) calc relevance score
      via pretrained CrossEncoder
    - sort shortlist by that score
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def _build_job_text_from_row(self, row: pd.Series) -> str:
        """
        Build vacancy text in a more structured way,
        so that cross-encoder can see seperate parts better.
        """
        title = str(row.get("title", ""))
        skills = str(row.get("skills", ""))
        description = str(row.get("description", ""))

        job_text = (
            f"Job title: {title}. "
            f"Skills: {skills}. "
            f"Description: {description}"
        )

        return clean_text(job_text)

    def rerank(self, resume_text: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank candidates returned by the retriever.
        """
        required_columns = ["title", "skills", "description", "score"]
        missing_columns = [col for col in required_columns if col not in candidates_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required candidate columns: {missing_columns}")

        results = candidates_df.copy()
        results["retrieval_score"] = results["score"]

        cleaned_resume = clean_text(resume_text)

        pairs = []
        for _, row in results.iterrows():
            job_text = self._build_job_text_from_row(row)
            pairs.append((cleaned_resume, job_text))

        reranker_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # predict может вернуть numpy array; превращаем в обычный столбец
        results["reranker_score"] = reranker_scores
        results = results.sort_values("reranker_score", ascending=False)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "reranker_score",
            "retrieval_score",
            "description",
            "skills",
        ]

        if "matched_skills" in results.columns:
            return_columns.append("matched_skills")

        if "role_family" in results.columns:
            return_columns.append("role_family")

        return results[return_columns]
