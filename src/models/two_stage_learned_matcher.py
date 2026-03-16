import pandas as pd

from src.models.embedding_matcher import EmbeddingJobMatcher
from src.models.learned_reranker import LearnedReranker

class TwoStageLearnedMatcher:
    """
    Two-stage architecture:
    1. Embedding retriever
    2. Learned reranker
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-macro-MiniLM-L6-v2",
        retriever_batch_size: int = 32,
        cross_encoder_batch_size: int = 16,
        cross_encoder_max_length: int = 512,
    ):
        self.retriever = EmbeddingJobMatcher(
            model_name=embedding_model_name,
            batch_size=retriever_batch_size
        )

        self.reranker = LearnedReranker(
            cross_encoder_model_name=cross_encoder_model_name,
            cross_encoder_batch_size=cross_encoder_batch_size,
            cross_encoder_max_length=cross_encoder_max_length
        )

    def fit_jobs(self, jobs_df: pd.DataFrame) -> None:
        self.retriever.fit(jobs_df)

    def fit_reranker(self, pair_df: pd.DataFrame) -> None:
        self.reranker.fit_on_pairs(pair_df)

    def recommend(
        self,
        resume_text: str,
        retrieve_top_k: int = 10,
        final_top_k: int = 5
    ):
        candidates = self.retriever.recommend(resume_text, top_k=retrieve_top_k)
        reranked = self.reranker.rerank(resume_text, candidates)
        return reranked.head(final_top_k)
