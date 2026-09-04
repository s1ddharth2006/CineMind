"""
src/recommender.py
Core Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource
except ImportError:
    def cache_resource_decorator(func):
        return func

class ContentRecommender:
    """
    Content-Based Movie Recommender powered by Scikit-Learn TF-IDF
    and Cosine Similarity.
    """

    def __init__(self, model_path: str = "models/similarity.pkl", data_path: str = "data/movies.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.movies_df: Optional[pd.DataFrame] = None
        self.top_similarities: Dict[int, List] = {}
        self.title_to_index: Dict[str, int] = {}
        self.id_to_index: Dict[int, int] = {}
        self.tfidf_vectorizer = None
        self.vocab_size: int = 0
        self.num_movies: int = 0
        self.is_ready: bool = False

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads precomputed similarity artifacts and movies dataset."""
        # Resolve path relative to project root
        base_dir = os.path.dirname(os.path.dirname(__file__))
        resolved_model = self.model_path if os.path.exists(self.model_path) else os.path.join(base_dir, self.model_path)
        resolved_data = self.data_path if os.path.exists(self.data_path) else os.path.join(base_dir, self.data_path)

        if not os.path.exists(resolved_data):
            raise FileNotFoundError(f"Movie data missing at {resolved_data}. Run scripts/build_model.py first.")

        # Load movies DataFrame
        self.movies_df = pd.read_csv(resolved_data)
        self.movies_df["movie_id"] = self.movies_df["movie_id"].astype(int)
        self.movies_df["title"] = self.movies_df["title"].fillna("")
        self.movies_df["year"] = self.movies_df["release_date"].fillna("").astype(str).str.slice(0, 4)

        # Load similarity model
        if os.path.exists(resolved_model):
            with open(resolved_model, "rb") as f:
                payload = pickle.load(f)
                self.top_similarities = payload.get("top_similarities", {})
                self.title_to_index = payload.get("title_to_index", {})
                self.id_to_index = payload.get("id_to_index", {})
                self.tfidf_vectorizer = payload.get("tfidf_vectorizer")
                self.vocab_size = payload.get("vocab_size", 5000)
                self.num_movies = payload.get("num_movies", len(self.movies_df))
            self.is_ready = True
        else:
            raise FileNotFoundError(f"Similarity model missing at {resolved_model}. Run scripts/build_model.py.")

    def resolve_movie_index(self, identifier) -> Optional[int]:
        """Resolves a title or movie_id to the internal DataFrame index."""
        if identifier is None:
            return None

        # Check if identifier is movie_id (int or numeric str)
        if isinstance(identifier, (int, np.integer)):
            return self.id_to_index.get(int(identifier))
        if isinstance(identifier, str) and identifier.isdigit():
            idx = self.id_to_index.get(int(identifier))
            if idx is not None:
                return idx

        # Lookup by title
        cleaned_title = str(identifier).strip().lower()
        if cleaned_title in self.title_to_index:
            return self.title_to_index[cleaned_title]

        # Fuzzy fallback: prefix search
        for title_key, idx in self.title_to_index.items():
            if title_key.startswith(cleaned_title) or cleaned_title in title_key:
                return idx

        return None

    def get_recommendations(self, identifier, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Calculates and returns the top N most similar movies to the given movie.

        Args:
            identifier: movie title (str) or TMDB movie_id (int)
            top_n: number of recommendations to return (default 10)

        Returns:
            List of movie dictionaries with similarity metrics and metadata.
        """
        if not self.is_ready:
            self._load_artifacts()

        idx = self.resolve_movie_index(identifier)
        if idx is None:
            return []

        # Retrieve precomputed similarities
        similar_items = self.top_similarities.get(idx, [])
        recommendations = []

        for sim_idx, score in similar_items[:top_n]:
            if sim_idx >= len(self.movies_df):
                continue
            movie_row = self.movies_df.iloc[sim_idx].to_dict()

            # Normalize similarity score to a clean percentage (e.g. 0.874 -> 87)
            # Apply slight non-linear scaling for better user-facing match percentages
            match_pct = max(15, min(99, int(score * 100)))

            movie_row["similarity_score"] = float(score)
            movie_row["match_percentage"] = match_pct
            movie_row["explanation"] = self.explain_recommendation(idx, sim_idx)
            recommendations.append(movie_row)

        return recommendations

    def explain_recommendation(self, base_idx: int, target_idx: int) -> Dict[str, Any]:
        """
        Provides explainable ML insights into why the target movie was recommended.
        Compares shared genres, shared director, shared actors, and keyword overlap.
        """
        base_movie = self.movies_df.iloc[base_idx]
        target_movie = self.movies_df.iloc[target_idx]

        # Shared genres
        base_genres = set(str(base_movie.get("genres", "")).split())
        target_genres = set(str(target_movie.get("genres", "")).split())
        shared_genres = list(base_genres.intersection(target_genres))

        # Shared director
        base_director = str(base_movie.get("director", "")).strip()
        target_director = str(target_movie.get("director", "")).strip()
        same_director = (base_director and target_director and base_director.lower() == target_director.lower())

        # Shared cast
        base_cast = set([c.lower() for c in str(base_movie.get("cast", "")).split()])
        target_cast = set([c.lower() for c in str(target_movie.get("cast", "")).split()])
        shared_cast = [c.capitalize() for c in base_cast.intersection(target_cast) if len(c) > 2]

        explanation_parts = []
        if same_director:
            explanation_parts.append(f"Directed by {base_director}")
        if shared_genres:
            explanation_parts.append(f"Shared genres: {', '.join(shared_genres[:3])}")
        if shared_cast:
            explanation_parts.append(f"Starring: {', '.join(shared_cast[:2])}")
        if not explanation_parts:
            explanation_parts.append("Thematic & plot similarity in story and keywords")

        return {
            "summary": " • ".join(explanation_parts),
            "shared_genres": shared_genres,
            "same_director": same_director,
            "director": base_director if same_director else None,
            "shared_cast": shared_cast
        }

@cache_resource_decorator
def get_recommender() -> ContentRecommender:
    """Factory helper to obtain a cached singleton instance of ContentRecommender."""
    return ContentRecommender()
