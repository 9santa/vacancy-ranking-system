import pandas as pd

from src.models.embedding_matcher import EmbeddingJobMatcher
from src.models.cross_encoder_reranker import CrossEncoderReranker


class TwoStageCrossEncoderMatcher:
    """
    Two-stage architecture:
    1. Embedding retriever -> top-k candidates
    2. Cross-encoder reranker -> final ranking
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        retriever_batch_size: int = 32,
        reranker_batch_size: int = 16,
        reranker_max_length: int = 512,
    ):
        self.retriever = EmbeddingJobMatcher(
            model_name=embedding_model_name,
            batch_size=retriever_batch_size,
        )
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
        )

    def fit(self, jobs_df: pd.DataFrame) -> None:
        self.retriever.fit(jobs_df)

    def recommend(
        self,
        resume_text: str,
        retrieve_top_k: int = 10,
        final_top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Step 1: retrieve top-N candidates with embeddings
        Step 2: rerank them with cross-encoder
        """
        candidates = self.retriever.recommend(resume_text, top_k=retrieve_top_k)
        reranked = self.reranker.rerank(resume_text, candidates)

        return reranked.head(final_top_k)
