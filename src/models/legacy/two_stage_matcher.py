import pandas as pd

from src.models.embedding_matcher import EmbeddingJobMatcher
from src.models.feature_reranker import FeatureBasedReranker

class TwoStageJobMatcher:
    """
    Two-stage architecture:
    1. Embedding retriever -> top-k candidates
    2. Feature-based reranker -> final ranking
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        retrieval_weight: float = 0.55,
        skill_weight: float = 0.20,
        domain_weight: float = 0.15,
        title_weight: float = 0.10
    ):
        self.retriever = EmbeddingJobMatcher(
            model_name=embedding_model_name,
            batch_size=batch_size
        )
        self.reranker = FeatureBasedReranker(
            retrieval_weight=retrieval_weight,
            skill_weight=skill_weight,
            domain_weight=domain_weight,
            title_weight=title_weight
        )

    def fit(self, jobs_df: pd.DataFrame) -> None:
        self.retriever.fit(jobs_df)

    def recommend(
        self,
        resume_text: str,
        retrieve_top_k: int = 10,
        final_top_k: int = 5
    ) -> pd.DataFrame:

        candidates = self.retriever.recommend(resume_text, top_k=retrieve_top_k)
        reranked = self.reranker.rerank(resume_text, candidates)

        return reranked.head(final_top_k)
