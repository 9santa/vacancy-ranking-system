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

        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()

        results = self.jobs_df.copy()
        results["score"] = similarities
        results = results.sort_values("score", ascending=False).head(top_k)

        return results[["job_id", "title", "company", "location", "score", "description", "skills"]]
