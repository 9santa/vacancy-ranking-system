import pandas as pd

from src.models.feature_reranker import FeatureBasedReranker
from src.models.cross_encoder_reranker import CrossEncoderReranker


class HybridNeuralReranker:
    """
    Hybrid neural reranker over top-k candidates.

    Combines:
    - retrieval score from embedding retriever
    - cross-encoder relevance score
    - feature-based bonuses

    The goal is to keep:
    - semantic strength of neural reranking
    - domain-specific precision of manual features
    """

    def __init__(
        self,
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        cross_encoder_batch_size: int = 16, # how many (resume, job) pairs the model scores at once
        cross_encoder_max_length: int = 512, # max total tokens after tokenization per pair. 512 is a common default for BERT.
        retrieval_weight: float = 0.20,
        cross_encoder_weight: float = 0.45,
        skill_weight: float = 0.20,
        domain_weight: float = 0.10,
        title_weight: float = 0.05,
    ):
        total_weight = (
            retrieval_weight
            + cross_encoder_weight
            + skill_weight
            + domain_weight
            + title_weight
        )
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError(
                f"Hybrid neural reranker weights must sum to 1.0, got {total_weight:.3f}"
            )

        self.retrieval_weight = retrieval_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.skill_weight = skill_weight
        self.domain_weight = domain_weight
        self.title_weight = title_weight

        self.cross_encoder_reranker = CrossEncoderReranker(
            model_name=cross_encoder_model_name,
            batch_size=cross_encoder_batch_size,
            max_length=cross_encoder_max_length,
        )

        self.feature_reranker = FeatureBasedReranker(
            retrieval_weight=0.55,
            skill_weight=0.20,
            domain_weight=0.15,
            title_weight=0.10,
        )

    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        min_score = scores.min()
        max_score = scores.max()

        if abs(max_score - min_score) < 1e-12:
            return pd.Series([1.0] * len(scores), index=scores.index)

        return (scores - min_score) / (max_score - min_score)

    def rerank(self, resume_text: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank embedding candidates using:
        - retrieval score
        - cross-encoder score
        - feature bonuses
        """
        required_columns = ["title", "skills", "description", "score"]
        missing_columns = [col for col in required_columns if col not in candidates_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required candidate columns: {missing_columns}")

        results = candidates_df.copy()
        results["retrieval_score"] = results["score"]

        # 1. Feature-based signals
        feature_results = self.feature_reranker.rerank(resume_text, results)

        feature_cols = [
            "job_id",
            "skill_overlap_bonus",
            "domain_phrase_bonus",
            "title_alignment_bonus",
            "matched_skills",
            "matched_domain_terms",
            "inferred_resume_family",
            "inferred_title_family",
        ]
        feature_results = feature_results[feature_cols]

        # 2. Cross-encoder scores
        ce_results = self.cross_encoder_reranker.rerank(resume_text, results)

        ce_cols = [
            "job_id",
            "reranker_score",
        ]
        ce_results = ce_results[ce_cols].rename(
            columns={"reranker_score": "cross_encoder_score"}
        )

        # 3. Merge all signals back
        if "matched_skills" in results.columns:
            results = results.drop(columns=["matched_skills"])

        results = results.merge(feature_results, on="job_id", how="left")
        results = results.merge(ce_results, on="job_id", how="left")

        # 4. Normalize numeric ranking signals inside shortlist
        results["retrieval_score_norm"] = self._normalize_scores(results["retrieval_score"])
        results["cross_encoder_score_norm"] = self._normalize_scores(results["cross_encoder_score"])

        # 5. Final hybrid score
        results["hybrid_score"] = (
            self.retrieval_weight * results["retrieval_score_norm"]
            + self.cross_encoder_weight * results["cross_encoder_score_norm"]
            + self.skill_weight * results["skill_overlap_bonus"]
            + self.domain_weight * results["domain_phrase_bonus"]
            + self.title_weight * results["title_alignment_bonus"]
        )

        results = results.sort_values("hybrid_score", ascending=False)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "hybrid_score",
            "retrieval_score",
            "retrieval_score_norm",
            "cross_encoder_score",
            "cross_encoder_score_norm",
            "skill_overlap_bonus",
            "domain_phrase_bonus",
            "title_alignment_bonus",
            "matched_skills",
            "matched_domain_terms",
            "inferred_resume_family",
            "inferred_title_family",
            "description",
            "skills",
        ]

        if "role_family" in results.columns:
            return_columns.append("role_family")

        return results[return_columns]
