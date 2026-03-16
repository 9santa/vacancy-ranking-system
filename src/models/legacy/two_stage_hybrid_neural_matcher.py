import pandas as pd

from src.models.embedding_matcher import EmbeddingJobMatcher
from src.models.hybrid_neural_reranker import HybridNeuralReranker


class TwoStageHybridNeuralMatcher:
    """
    Two-stage architecture:
    1. Embedding retriever
    2. Hybrid neural reranker

    Hybrid neural reranker combines:
    - retrieval score
    - cross-encoder score
    - feature-based bonuses
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size: int = 32,
        cross_encoder_batch_size: int = 16,
        cross_encoder_max_length: int = 512,
        retrieval_weight: float = 0.20,
        cross_encoder_weight: float = 0.45,
        skill_weight: float = 0.20,
        domain_weight: float = 0.10,
        title_weight: float = 0.05,
    ):
        self.retriever = EmbeddingJobMatcher(
            model_name=embedding_model_name,
            batch_size=retriever_batch_size,
        )

        self.reranker = HybridNeuralReranker(
            cross_encoder_model_name=cross_encoder_model_name,
            cross_encoder_batch_size=cross_encoder_batch_size,
            cross_encoder_max_length=cross_encoder_max_length,
            retrieval_weight=retrieval_weight,
            cross_encoder_weight=cross_encoder_weight,
            skill_weight=skill_weight,
            domain_weight=domain_weight,
            title_weight=title_weight,
        )

    def fit(self, jobs_df: pd.DataFrame) -> None:
        self.retriever.fit(jobs_df)

    def recommend(
        self,
        resume_text: str,
        retrieve_top_k: int = 10,
        final_top_k: int = 5,
    ) -> pd.DataFrame:
        candidates = self.retriever.recommend(resume_text, top_k=retrieve_top_k)
        reranked = self.reranker.rerank(resume_text, candidates)

        return reranked.head(final_top_k)
