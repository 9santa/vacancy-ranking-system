from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cross_encoder_reranker import CrossEncoderReranker
from src.models.feature_reranker import FeatureBasedReranker


class LearnedReranker:
    """
    Pointwise learned reranker.
    Uses:
    - retrieval score
    - cross-encoder score
    - feature-based bonuses

    and learns how to combine them from data.
    """

    FEATURE_COLUMNS = [
        "retrieval_score_norm",
        "cross_encoder_score_norm",
        "skill_overlap_bonus",
        # "domain_phrase_bonus",
        "title_alignment_bonus",
    ]

    def __init__(
        self,
        cross_encoder_model_name: str = "cross-encoder/ms-macro-MiniLM-L6-v2",
        cross_encoder_batch_size: int = 16,
        cross_encoder_max_length: int = 512,
    ):
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

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",  # classifier
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        min_score = scores.min()
        max_score = scores.max()

        if abs(max_score - min_score) < 1e-12:
            return pd.Series([1.0] * len(scores), index=scores.index)

        return (scores - min_score) / (max_score - min_score)

    def build_feature_frame(
        self, resume_text: str, candidates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Builds pairwise features for (resume, job) pairs.
        """
        required_columns = ["title", "skills", "description", "score"]
        missing_columns = [
            col for col in required_columns if col not in candidates_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required candidate columns: {missing_columns}")

        results = candidates_df.copy()
        results["retrieval_score"] = results["score"]

        if "matched_skills" in results.columns:
            results = results.drop(columns=["matched_skills"])

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

        ce_results = self.cross_encoder_reranker.rerank(resume_text, results)
        ce_results = ce_results[["job_id", "reranker_score"]].rename(
            columns={"reranker_score": "cross_encoder_score"}
        )

        results = results.merge(feature_results, on="job_id", how="left")
        results = results.merge(ce_results, on="job_id", how="left")

        results["retrieval_score_norm"] = self._normalize_scores(
            results["retrieval_score"]
        )
        results["cross_encoder_score_norm"] = self._normalize_scores(
            results["cross_encoder_score"]
        )

        return results

    def fit_on_pairs(self, pair_df: pd.DataFrame) -> None:
        missing_columns = [
            col
            for col in self.FEATURE_COLUMNS + ["label"]
            if col not in pair_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for training: {missing_columns}"
            )

        X = pair_df[self.FEATURE_COLUMNS]
        y = pair_df["label"]

        self.model.fit(X, y)
        self.is_fitted = True

    def predict_scores(self, feature_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Learned reranker is not fitted.")

        X = feature_df[self.FEATURE_COLUMNS]
        probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=feature_df.index)

    def rerank(self, resume_text: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = self.build_feature_frame(resume_text, candidates_df)
        feature_df["learned_score"] = self.predict_scores(feature_df)
        feature_df = feature_df.sort_values("learned_score", ascending=False)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "learned_score",
            "retrieval_score",
            "cross_encoder_score",
            "retrieval_score_norm",
            "cross_encoder_score_norm",
            "skill_overlap_bonus",
            "domain_phrase_bonus",
            "title_alignment_bonus",
            "matched_skills",
            "matched_domain_terms",
            "description",
            "skills",
        ]

        if "role_family" in feature_df.columns:
            return_columns.append("role_family")

        if "role_subfamily" in feature_df.columns:
            return_columns.append("role_subfamily")

        return feature_df[return_columns]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self.is_fitted = True
