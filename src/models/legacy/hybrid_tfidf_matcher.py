import math
from numpy import ma
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaning import clean_text
from src.models.structured_tfidf_matcher import StructuredTfidfJobMatcher


"""
New Score Calculation:
final_score = 0.70 * structured_score +
              0.15 * skill_overlap_bonus +
              0.10 * domain_phrase_bonus +
              0.05 * family_alignment_bonus
Where:
structured_score - current structured TF-IDF
skill_overlap_bonus - how well vacancy skills cover resume
domain_phrase_bonus - how well important phrases like 'power bi', 'ab testing' match
family_alignment_bonus - does inferred role match resume text
"""

class HybridTfidfJobMatcher(StructuredTfidfJobMatcher):
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
        title_weight: float = 0.45,
        skills_weight: float = 0.35,
        description_weight: float = 0.20,
        base_score_weight: float = 0.70,
        skill_bonus_weight: float = 0.15,
        domain_bonus_weight: float = 0.10,
        family_bonus_weight: float = 0.05,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2)
    ):
        super().__init__(
            title_weight=title_weight,
            skills_weight=skills_weight,
            description_weight=description_weight,
            max_features=max_features,
            ngram_range=ngram_range
        )

        total_bonus_weight = (
            base_score_weight
            + skill_bonus_weight
            + domain_bonus_weight
            + family_bonus_weight
        )
        if abs(total_bonus_weight - 1.0) > 1e-8:
            raise ValueError(f"Hybrid weight must sum to 1.0, got {total_bonus_weight:.3f}")

        self.base_score_weight = base_score_weight
        self.skill_bonus_weight = skill_bonus_weight
        self.domain_bonus_weight = domain_bonus_weight
        self.family_bonus_weight = family_bonus_weight

        # Normalize phrases via clean_text(), so that comparasion is fair
        self.role_phrase_weights = {}
        for family, phrase_dict in self.ROLE_PHRASE_WEIGHTS.items():
            self.role_phrase_weights[family] = {
                clean_text(phrase): weight for phrase, weight in phrase_dict.items()
            }

    def _clean_and_pad_text(self, text: str) -> str:
        """
        Clean text + pad spaces around it
        """
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
        if len(job_skills) == 0:
            return 0.0, []

        matched_skills = self._get_matched_skills(resume_text, skills_text)
        bonus = len(matched_skills) / len(job_skills)

        return (bonus, matched_skills)

    # Returns set of phrase signals found in text
    def _extract_matched_phrases(
        self,
        text: str,
        phrase_weights: dict[str, float]
    ) -> set[str]:

        padded_text = self._clean_and_pad_text(text)
        matched = set()
        for phrase in phrase_weights.keys():
            if f" {phrase} " in padded_text:
                matched.add(phrase)

        return matched

    def _get_family_signal_scores(self, text: str) -> dict[str, float]:
        """
        Calculates signal score for each role family.
        For example, if text has:
        'power bi', 'reporting', 'kpis'
        then score for family 'bi' will be large.
        """
        scores = {}

        for family, phrase_weights in self.role_phrase_weights.items():
            matched_phrases = self._extract_matched_phrases(text, phrase_weights)
            score = sum(phrase_weights[phrase] for phrase in matched_phrases)
            scores[family] = score

        return scores

    def _infer_role_family_from_text(self, text: str) -> str | None:
        """
        Simple rule-based inference:
        choose family with max phrase score.
        If signals not found, return None.
        """
        scores = self._get_family_signal_scores(text)
        best_family = max(scores, key=scores.get)

        if scores[best_family] <= 0:
            return None

        return best_family

    def _get_domain_phrase_bonus(
        self,
        resume_text: str,
        job_text: str
    ) -> tuple[float, list[str]]:

        all_resume_matches = set()
        all_job_matches = set()
        all_phrases_weights = {}

        for family, phrase_weights in self.role_phrase_weights.items():
            resume_matches = self._extract_matched_phrases(resume_text, phrase_weights)
            job_matches = self._extract_matched_phrases(job_text, phrase_weights)

            all_resume_matches.update(resume_matches)
            all_job_matches.update(job_matches)
            all_phrases_weights.update(phrase_weights)

        if len(all_job_matches) == 0:
            return (0.0, [])

        shared_matches = all_resume_matches.intersection(all_job_matches)

        matched_weight = sum(all_phrases_weights[phrase] for phrase in shared_matches)
        job_present_weight = sum(all_phrases_weights[phrase] for phrase in all_job_matches)

        bonus = matched_weight / job_present_weight
        return (bonus, sorted(shared_matches))

    def _get_family_alignment_bonus(
        self,
        resume_text: str,
        job_text: str
    ) -> tuple[float, str | None, str | None]:
        """
        Bonus = 1.0, if rule-based inferred family matches
        Else 0.0
        """

        inferred_resume_family = self._infer_role_family_from_text(resume_text)
        inferred_job_family = self._infer_role_family_from_text(job_text)

        if inferred_resume_family is None or inferred_job_family is None:
            return (0.0, inferred_resume_family, inferred_job_family)

        bonus = float(inferred_resume_family == inferred_job_family)
        return (bonus, inferred_resume_family, inferred_job_family)

    def recommend(self, resume_text: str, top_k: int = 10) -> pd.DataFrame:
        if self.jobs_df is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        cleaned_resume = clean_text(resume_text)

        # Structured TF-IDF part
        resume_title_vector = self.title_vectorizer.transform([cleaned_resume])
        resume_skills_vector = self.skills_vectorizer.transform([cleaned_resume])
        resume_description_vector = self.description_vectorizer.transform([cleaned_resume])

        title_sim = cosine_similarity(resume_title_vector, self.title_vectors).flatten()
        skills_sim = cosine_similarity(resume_skills_vector, self.skills_vectors).flatten()
        description_sim = cosine_similarity(
            resume_description_vector, self.description_vectors
        ).flatten()

        structured_score = (
            self.title_weight * title_sim
            + self.skills_weight * skills_sim
            + self.description_weight * description_sim
        )

        results = self.jobs_df.copy()

        results["title_score"] = title_sim
        results["skills_score"] = skills_sim
        results["description_score"] = description_sim
        results["structured_score"] = structured_score

        matched_skills_list = []
        skill_bonus_list = []
        domain_bonus_list = []
        family_bonus_list = []
        total_bonus_list = []
        inferred_resume_family_list = []
        inferred_job_family_list = []
        matched_domain_terms_list = []

        title_overlap_terms_list = []
        skills_overlap_terms_list = []
        description_overlap_terms_list = []

        for idx, row in results.iterrows():
            job_text_for_rules = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("skills", "")),
                    str(row.get("description", "")),
                ]
            )

            # Bonus 1: exact phrase skill overlap
            skill_overlap_bonus, matched_skills = self._get_skill_overlap_bonus(
                resume_text,
                row["skills"],
            )

            # Bonus 2: role-specific phrase overlap
            domain_phrase_bonus, matched_domain_terms = self._get_domain_phrase_bonus(
                resume_text,
                job_text_for_rules,
            )

            # Bonus 3: inferred family alignment
            family_alignment_bonus, inferred_resume_family, inferred_job_family = (
                self._get_family_alignment_bonus(resume_text, job_text_for_rules)
            )

            total_bonus = (
                self.skill_bonus_weight * skill_overlap_bonus
                + self.domain_bonus_weight * domain_phrase_bonus
                + self.family_bonus_weight * family_alignment_bonus
            )

            # Structured explainability
            title_overlap_terms = self._get_top_overlap_terms(
                resume_title_vector,
                self.title_vectors[idx],
                self.title_vectorizer,
                top_n=3,
            )
            skills_overlap_terms = self._get_top_overlap_terms(
                resume_skills_vector,
                self.skills_vectors[idx],
                self.skills_vectorizer,
                top_n=3,
            )
            description_overlap_terms = self._get_top_overlap_terms(
                resume_description_vector,
                self.description_vectors[idx],
                self.description_vectorizer,
                top_n=5,
            )

            matched_skills_list.append(", ".join(matched_skills))
            skill_bonus_list.append(skill_overlap_bonus)
            domain_bonus_list.append(domain_phrase_bonus)
            family_bonus_list.append(family_alignment_bonus)
            total_bonus_list.append(total_bonus)
            inferred_resume_family_list.append(inferred_resume_family)
            inferred_job_family_list.append(inferred_job_family)
            matched_domain_terms_list.append(", ".join(matched_domain_terms))

            title_overlap_terms_list.append(", ".join(title_overlap_terms))
            skills_overlap_terms_list.append(", ".join(skills_overlap_terms))
            description_overlap_terms_list.append(", ".join(description_overlap_terms))

        results["matched_skills"] = matched_skills_list
        results["skill_overlap_bonus"] = skill_bonus_list
        results["domain_phrase_bonus"] = domain_bonus_list
        results["family_alignment_bonus"] = family_bonus_list
        results["bonus_score"] = total_bonus_list
        results["inferred_resume_family"] = inferred_resume_family_list
        results["inferred_job_family"] = inferred_job_family_list
        results["matched_domain_terms"] = matched_domain_terms_list

        results["title_overlap_terms"] = title_overlap_terms_list
        results["skills_overlap_terms"] = skills_overlap_terms_list
        results["description_overlap_terms"] = description_overlap_terms_list

        results["score"] = (
            self.base_score_weight * results["structured_score"]
            + results["bonus_score"]
        )

        results = results.sort_values("score", ascending=False).head(top_k)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "score",
            "structured_score",
            "bonus_score",
            "title_score",
            "skills_score",
            "description_score",
            "skill_overlap_bonus",
            "domain_phrase_bonus",
            "family_alignment_bonus",
            "matched_skills",
            "matched_domain_terms",
            "inferred_resume_family",
            "inferred_job_family",
            "title_overlap_terms",
            "skills_overlap_terms",
            "description_overlap_terms",
            "description",
            "skills",
        ]

        if "role_family" in results.columns:
            return_columns.append("role_family")

        return results[return_columns]







