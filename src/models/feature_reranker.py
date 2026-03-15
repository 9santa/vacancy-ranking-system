import pandas as pd

from src.preprocessing.text_cleaning import clean_text

class FeatureBasedReranker:
    """
    Lightweight reranker over top-k candidates.

    Works over embedding retriever.
    """

    ROLE_PHRASE_WEIGHTS = {
        "data_science": {
        "data scientist": 1.0,
        "data science": 0.8,
        "machine learning": 1.0,
        "scikit learn": 1.0,
        "forecasting": 0.9,
        "nlp": 0.9,
        "feature engineering": 1.0,
        "regression": 0.8,
        "classification": 0.8,
        "predictive": 0.8,
        "model evaluation": 0.9,
        },
        "analytics": {
            "data analyst": 0.8,
            "analytics": 0.6,
            "product analyst": 1.0,
            "product analytics": 1.0,
            "ab testing": 1.0,
            "a b testing": 1.0,
            "retention": 0.8,
            "funnel": 0.8,
            "segmentation": 0.8,
            "user behavior": 0.8,
            "campaign performance": 1.0,
            "marketing analytics": 1.0,
        },
        "bi": {
            "bi analyst": 1.0,
            "business intelligence": 1.0,
            "power bi": 1.0,
            "reporting": 0.8,
            "reporting analyst": 1.0,
            "dashboard": 0.4,
            "dashboards": 0.4,
            "kpi": 0.4,
            "kpis": 0.4,
        },
    }

    def __init__(
        self,
        retrieval_weight: float = 0.55,
        skill_weight: float = 0.20,
        domain_weight: float = 0.15,
        title_weight: float = 0.10,
    ):
        total_weight = retrieval_weight + skill_weight + domain_weight + title_weight
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError(f"Reranker weights must sum to 1.0, got: {total_weight}")

        self.retrieval_weight = retrieval_weight
        self.skill_weight = skill_weight
        self.domain_weight = domain_weight
        self.title_weight = title_weight

        self.role_phrase_weights = {}
        for family, phrase_dict in self.ROLE_PHRASE_WEIGHTS.items():
            self.role_phrase_weights[family] = {
                clean_text(phrase): weight for phrase, weight in phrase_dict.items()
            }

    def _clean_and_pad_text(self, text: str) -> str:
        cleaned = clean_text(text)
        return f" {cleaned} "


    def _extract_job_skills(self, skills_text: str) -> list[str]:
        if not isinstance(skills_text, str):
            return []

        skills = [skill.strip() for skill in skills_text.split(";")]
        normalized_skills = [clean_text(skill) for skill in skills]
        return normalized_skills

    def _get_matched_skills(self, resume_text: str, skills_text: str) -> list[str]:
        padded_resume = self._clean_and_pad_text(resume_text)
        job_skills = self._extract_job_skills(skills_text)

        matched = []
        for skill in job_skills:
            if f" {skill} " in padded_resume:
                matched.append(skill)

        return sorted(set(matched))

    def _get_skill_overlap_bonus(self, resume_text: str, skills_text: str) -> tuple[float, list[str]]:
        job_skills = self._extract_job_skills(skills_text)
        if not job_skills:
            return 0.0, []

        matched_skills = self._get_matched_skills(resume_text, skills_text)
        bonus = len(matched_skills) / len(job_skills)
        return bonus, matched_skills

    def _extract_matched_phrases(self, text: str, phrase_weights: dict[str, float]) -> set[str]:
        padded_text = self._clean_and_pad_text(text)

        matched = set()
        for phrase in phrase_weights.keys():
            if f" {phrase} " in padded_text:
                matched.add(phrase)

        return matched

    def _get_family_signal_scores(self, text: str) -> dict[str, float]:
        scores = {}

        for family, phrase_weights in self.role_phrase_weights.items():
            matched_phrases = self._extract_matched_phrases(text, phrase_weights)
            score = sum(phrase_weights[phrase] for phrase in matched_phrases)
            scores[family] = score


        return scores

    def _infer_role_family_from_text(self, text: str) -> str | None:
        scores = self._get_family_signal_scores(text)
        best_family = max(scores, key=scores.get)

        if scores[best_family] <= 0:
            return None

        return best_family

    def _get_domain_phrase_bonus(self, resume_text: str, job_text: str) -> tuple[float, list[str]]:
        all_resume_matches = set()
        all_job_matches = set()
        all_phrase_weights = {}

        for family, phrase_weights in self.role_phrase_weights.items():
            resume_matches = self._extract_matched_phrases(resume_text, phrase_weights)
            job_matches = self._extract_matched_phrases(job_text, phrase_weights)

            all_resume_matches.update(resume_matches)
            all_job_matches.update(job_matches)
            all_phrase_weights.update(phrase_weights)

        if not all_job_matches:
            return 0.0, []

        shared_matches = all_resume_matches.intersection(all_job_matches)

        matched_weight = sum(all_phrase_weights[phrase] for phrase in shared_matches)
        job_present_weight = sum(all_phrase_weights[phrase] for phrase in all_job_matches)

        bonus = matched_weight / job_present_weight
        return bonus, sorted(shared_matches)

    def _get_title_alignment_bonus(
        self,
        resume_text: str,
        title_text: str
    ) -> tuple[float, str | None, str | None]:
        inferred_resume_family = self._infer_role_family_from_text(resume_text)
        inferred_title_family = self._infer_role_family_from_text(title_text)

        if inferred_resume_family is None or inferred_title_family is None:
            return 0.0, inferred_resume_family, inferred_title_family

        bonus = float(inferred_resume_family == inferred_title_family)
        return bonus, inferred_resume_family, inferred_title_family

    # minmax normalization, because scores can be very closely grouped together
    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        min_score = scores.min()
        max_score = scores.max()

        if abs(max_score - min_score) < 1e-12:
            return pd.Series([1.0] * len(scores), index=scores.index)

        return (scores - min_score) / (max_score - min_score)

    def rerank(self, resume_text: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["title", "skills", "description", "score"]
        missing_columns = [col for col in required_columns if col not in candidates_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required candidate columns: {missing_columns}")

        results = candidates_df.copy()
        results["retrieval_score"] = results["score"]
        results["retrieval_score_norm"] = self._normalize_scores(results["retrieval_score"])

        skill_bonus_list = []
        domain_bonus_list = []
        title_bonus_list = []
        matched_skills_list = []
        matched_domain_terms_list = []
        inferred_resume_family_list = []
        inferred_title_family_list = []

        for _, row in results.iterrows():
            title_text = str(row.get("title", ""))
            skills_text = str(row.get("skills", ""))
            description_text = str(row.get("description", ""))

            job_text_for_rules = " ".join([title_text, skills_text, description_text])

            skill_bonus, matched_skills = self._get_skill_overlap_bonus(
                resume_text,
                skills_text,
            )

            domain_bonus, matched_domain_terms = self._get_domain_phrase_bonus(
                resume_text,
                job_text_for_rules,
            )

            title_bonus, inferred_resume_family, inferred_title_family = (
                self._get_title_alignment_bonus(resume_text, title_text)
            )

            skill_bonus_list.append(skill_bonus)
            domain_bonus_list.append(domain_bonus)
            title_bonus_list.append(title_bonus)
            matched_skills_list.append(", ".join(matched_skills))
            matched_domain_terms_list.append(", ".join(matched_domain_terms))
            inferred_resume_family_list.append(inferred_resume_family)
            inferred_title_family_list.append(inferred_title_family)

        results["skill_overlap_bonus"] = skill_bonus_list
        results["domain_phrase_bonus"] = domain_bonus_list
        results["title_alignment_bonus"] = title_bonus_list
        results["matched_skills"] = matched_skills_list
        results["matched_domain_terms"] = matched_domain_terms_list
        results["inferred_resume_family"] = inferred_resume_family_list
        results["inferred_title_family"] = inferred_title_family_list

        results["rerank_score"] = (
            self.retrieval_weight * results["retrieval_score_norm"]
            + self.skill_weight * results["skill_overlap_bonus"]
            + self.domain_weight * results["domain_phrase_bonus"]
            + self.title_weight * results["title_alignment_bonus"]
        )

        results = results.sort_values("rerank_score", ascending=False)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "rerank_score",
            "retrieval_score",
            "retrieval_score_norm",
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
