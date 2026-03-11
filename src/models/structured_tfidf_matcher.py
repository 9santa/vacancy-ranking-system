import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaning import clean_text

class StructuredTfidfJobMatcher:
    def __init__(self,
                 title_weight: float = 0.45,
                 skills_weight: float = 0.35,
                 description_weight: float = 0.20,
                 max_features: int = 5000,
                 ngram_range: tuple = (1,2)):

        total_weight = title_weight + skills_weight + description_weight
        # Check for weights correctness
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.3f}")

        self.title_weight = title_weight
        self.skills_weight = skills_weight
        self.description_weight = description_weight

        self.title_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.skills_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.description_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )

        self.title_vectors = None
        self.skills_vectors = None
        self.description_vectors = None

        self.jobs_df = None

    def fit(self, jobs_df: pd.DataFrame) -> None:
        """
        Fit three different IF-IDF spaces:
        - title
        - skills
        - description
        """

        required_columns = ["title", "skills", "description"]
        missing_columns = [col for col in required_columns if col not in jobs_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for structured matcher: {missing_columns}")

        self.jobs_df = jobs_df.copy()

        for col in required_columns:
            self.jobs_df[col] = self.jobs_df[col].fillna(value="").astype(str)

        cleaned_titles = self.jobs_df["title"].apply(clean_text).tolist()
        cleaned_skills = self.jobs_df["skills"].apply(clean_text).tolist()
        cleaned_description = self.jobs_df["description"].apply(clean_text).tolist()

        self.title_vectors = self.title_vectorizer.fit_transform(cleaned_titles)
        self.skills_vectors = self.skills_vectorizer.fit_transform(cleaned_skills)
        self.description_vectors = self.description_vectorizer.fit_transform(cleaned_description)

    def _extract_resume_tokens(self, resume_text: str) -> set[str]:
        """
        Simple resume tokenization:
        clean text and transform into set of words.
        """
        cleaned_resume = clean_text(resume_text)
        return set(cleaned_resume.split())

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
        resume_tokens = self._extract_resume_tokens(resume_text)
        job_skills = self._extract_job_skills(skills_text)

        matched = sorted(job_skills.intersection(resume_tokens))
        return matched

    # Universal explainability helper for any field:
# title / skills / description
    def _get_top_overlap_terms(
        self,
        resume_vector,
        job_vector,
        vectorizer,
        top_n: int = 5,
    ) -> list[str]:

        overlap = resume_vector.multiply(job_vector)

        feature_names = vectorizer.get_feature_names_out()
        overlap_array = overlap.toarray().flatten()
        nonzero_indices = overlap_array.nonzero()[0]

        if len(nonzero_indices) == 0:
            return []

        sorted_indices = sorted(
            nonzero_indices,
            key=lambda idx: (overlap_array[idx], len(feature_names[idx].split())),
            reverse=True
        )

        generic_terms = {
            "data",
            "science",
            "project",
            "projects",
            "work",
            "worked",
            "experience",
            "student",
            "analysis",
        }

        selected_terms = []
        selected_word_sets = []

        for idx in sorted_indices:
            term = feature_names[idx].strip()
            words = term.split()

            # skip too generic one-word terms
            if term in generic_terms:
                continue

            word_set = set(words)

            # if already selected longer term with the same words,
            # skip the short one
            is_redundant = False
            for existing_set in selected_word_sets:
                if word_set.issubset(existing_set):
                    is_redundant = True
                    break

            if is_redundant:
                continue

            selected_terms.append(term)
            selected_word_sets.append(word_set)

            if len(selected_terms) == top_n:
                break

        return selected_terms

    # Returns top-k vacancies + score per field
    def recommend(self, resume_text: str, top_k: int = 10) -> pd.DataFrame:
        if self.jobs_df is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        cleaned_resume = clean_text(resume_text)

        resume_title_vector = self.title_vectorizer.transform([cleaned_resume])
        resume_skills_vector = self.skills_vectorizer.transform([cleaned_resume])
        resume_description_vector = self.description_vectorizer.transform([cleaned_resume])

        title_sim = cosine_similarity(resume_title_vector, self.title_vectors).flatten()
        skills_sim = cosine_similarity(resume_skills_vector, self.skills_vectors).flatten()
        description_sim = cosine_similarity(resume_description_vector, self.description_vectors).flatten()

        final_score = (
            self.title_weight * title_sim
            + self.skills_weight * skills_sim
            + self.description_weight * description_sim
        )

        results = self.jobs_df.copy()

        results["title_score"] = title_sim
        results["skills_score"] = skills_sim
        results["description_score"] = description_sim
        results["score"] = final_score

        matched_skills_list = []
        title_overlap_terms_list = []
        skills_overlap_terms_list = []
        description_overlap_terms_list = []

        for idx, row in results.iterrows():
            matched_skills = self._get_matched_skills(resume_text, row["skills"])

            title_overlap_terms = self._get_top_overlap_terms(
                resume_title_vector,
                self.title_vectors[idx],
                self.title_vectorizer,
                top_n=3
            )
            skills_overlap_terms = self._get_top_overlap_terms(
                resume_skills_vector,
                self.skills_vectors[idx],
                self.skills_vectorizer,
                top_n=3
            )
            description_overlap_terms = self._get_top_overlap_terms(
                resume_description_vector,
                self.description_vectors[idx],
                self.description_vectorizer,
                top_n=3
            )

            matched_skills_list.append(", ".join(matched_skills))
            title_overlap_terms_list.append(", ".join(title_overlap_terms))
            skills_overlap_terms_list.append(", ".join(skills_overlap_terms))
            description_overlap_terms_list.append(", ".join(description_overlap_terms))


        results["matched_skills"] = matched_skills_list
        results["title_overlap_terms"] = title_overlap_terms_list
        results["skills_overlap_terms"] = skills_overlap_terms_list
        results["description_overlap_terms"] = description_overlap_terms_list

        results = results.sort_values("score", ascending=False).head(top_k)

        return_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "score",
            "title_score",
            "skills_score",
            "description_score",
            "matched_skills",
            "title_overlap_terms",
            "skills_overlap_terms",
            "description_overlap_terms",
            "description",
            "skills",
        ]

        if "role_family" in results.columns:
            return_columns.append("role_family")

        return results[return_columns]


