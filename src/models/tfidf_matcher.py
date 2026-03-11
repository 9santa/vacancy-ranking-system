# TF-IDF doesn't understand 'meaning' of the words, it only catches text matches
# 'ml' and 'machine learning' might match badly, even though they mean the same thing
# It's lexical matching, not semantic matching

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaning import clean_text

# Simple baseline matcher:
# - vectorize vacancies with TF-IDF
# - vectorize resume same way
# - calculate cosine similarity
# - return top-k vacancies
class TfidfJobMatcher:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.job_vectors = None
        self.jobs_df = None

    def fit(self, jobs_df: pd.DataFrame) -> None:
        # train vectorizer on vacancies text
        self.jobs_df = jobs_df.copy()
        cleaned_jobs_texts = self.jobs_df["job_text"].apply(clean_text).tolist()
        # Builds a dictionary of terms
        # Calculates the importance of words based on a corpus of vacancies
        self.job_vectors = self.vectorizer.fit_transform(cleaned_jobs_texts)

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

    def _get_top_overlap_terms(self, resume_vector, job_index: int, top_n: int = 5) -> list[str]:
        """
        Find features TF-IDF, which matched the most between resume and some vacancy
        Idea:
        - resume has TF-IDF weights on words
        - vacancy has TF-IDF weights on the same words
        - multiply them per element
        - take top-N features with the highest contribution
        """
        job_vector = self.job_vectors[job_index]

        # Per element weight multiply
        overlap = resume_vector.multiply(job_vector)

        feature_names = self.vectorizer.get_feature_names_out()
        overlap_array = overlap.toarray().flatten()

        # Indices with non-zero contribution
        nonzero_indices = overlap_array.nonzero()[0]

        if len(nonzero_indices) == 0:
            return []

        # Sort by descending contribution
        sorted_indices = sorted(nonzero_indices, key=lambda idx: (overlap_array[idx], len(feature_names[idx].split())), reverse=True)

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



    # Returns top-k most relevant vacancies for this resume
    # 1) Clean text
    # 2) Vectorize into the same TF-IDF space
    # 3) Calculate cosine similarity with each vacancy
    # 4) Sort and return
    def recommend(self, resume_text: str, top_k: int = 10) -> pd.DataFrame:
        if self.job_vectors is None or self.jobs_df is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        cleaned_resume = clean_text(resume_text)
        resume_vector = self.vectorizer.transform([cleaned_resume])

        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten() # flatten() converts multi-dim array into 1D

        results = self.jobs_df.copy()
        results["score"] = similarities

        # For each vacancy calc explainability fields
        matched_skills_list = []
        overlap_terms_list = []

        for idx, row in results.iterrows():
            matched_skills = self._get_matched_skills(resume_text, row["skills"])
            overlap_terms = self._get_top_overlap_terms(resume_vector, idx, top_n=5)

            matched_skills_list.append(", ".join(matched_skills))
            overlap_terms_list.append(", ".join(overlap_terms))

        results["matched_skills"] = matched_skills_list
        results["overlap_terms"] = overlap_terms_list

        results = results.sort_values("score", ascending=False).head(top_k)

        return_cols = [
                "job_id",
                "title",
                "company",
                "location",
                "score",
                "matched_skills",
                "overlap_terms",
                "description",
                "skills",
        ]

        if "role_family" in results.columns:
            return_cols.append("role_family")

        return results[return_cols]


