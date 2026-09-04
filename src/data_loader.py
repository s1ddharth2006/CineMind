"""
src/data_loader.py
Data loading, indexing, and lookup functions for CineMind.
"""
import os
import re
import pandas as pd

# Optional Streamlit caching if running within Streamlit context
try:
    import streamlit as st
    cache_data_decorator = st.cache_data
except ImportError:
    def cache_data_decorator(func):
        return func

@cache_data_decorator
def load_movie_data(data_path: str = "data/movies.csv") -> pd.DataFrame:
    """
    Loads movies dataset, cleans missing values, and parses release years.
    """
    if not os.path.exists(data_path):
        # Handle relative path resolution if run from different working directory
        parent_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)
        if os.path.exists(parent_path):
            data_path = parent_path
        else:
            raise FileNotFoundError(f"Movie dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Clean text columns
    text_cols = ["title", "overview", "genres", "keywords", "cast", "director"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Extract release year
    def extract_year(date_val):
        if not isinstance(date_val, str):
            return "N/A"
        match = re.search(r"\b(19\d\d|20\d\d)\b", date_val)
        return match.group(0) if match else "N/A"

    df["year"] = df["release_date"].apply(extract_year)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0).round(1)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0).round(1)
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").fillna(0).astype(int)

    return df

def get_movie_by_id(df: pd.DataFrame, movie_id: int):
    """Retrieve movie record by TMDB movie_id."""
    match = df[df["movie_id"] == int(movie_id)]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

def get_movie_by_title(df: pd.DataFrame, title: str):
    """Retrieve movie record by exact or case-insensitive title."""
    if not title or not isinstance(title, str):
        return None
    cleaned = title.strip().lower()
    match = df[df["title"].str.lower() == cleaned]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

def search_movies(df: pd.DataFrame, query: str, limit: int = 15):
    """
    Search movies by title query:
    1. Exact match
    2. Prefix matches (starts with query)
    3. Contains query
    4. Keyword/director matches
    """
    if not query or not isinstance(query, str):
        return []

    q = query.strip().lower()
    if len(q) == 0:
        return []

    titles = df["title"].str.lower()

    # Exact match
    exact = df[titles == q]
    # Starts with query
    starts = df[titles.str.startswith(q) & (titles != q)]
    # Contains query
    contains = df[titles.str.contains(re.escape(q), regex=True, na=False) & ~titles.str.startswith(q)]

    combined = pd.concat([exact, starts, contains]).drop_duplicates(subset=["movie_id"])

    # If few results, check director or cast
    if len(combined) < limit:
        people_match = df[
            (df["director"].str.lower().str.contains(re.escape(q), regex=True, na=False) |
             df["cast"].str.lower().str.contains(re.escape(q), regex=True, na=False)) &
            ~df["movie_id"].isin(combined["movie_id"])
        ]
        combined = pd.concat([combined, people_match])

    return combined.head(limit).to_dict("records")

def get_popular_movies(df: pd.DataFrame, limit: int = 12):
    """Retrieve top trending / popular movies with good ratings."""
    filtered = df[(df["vote_count"] > 1500) & (df["vote_average"] >= 7.0)]
    if len(filtered) < limit:
        filtered = df.sort_values(by="popularity", ascending=False)
    else:
        filtered = filtered.sort_values(by=["popularity", "vote_average"], ascending=[False, False])
    return filtered.head(limit).to_dict("records")

def get_movies_by_genre(df: pd.DataFrame, genre_name: str, limit: int = 12):
    """Retrieve top rated movies matching a given genre."""
    if not genre_name:
        return []
    mask = df["genres"].str.contains(r"\b" + re.escape(genre_name) + r"\b", case=False, regex=True, na=False)
    genre_movies = df[mask].sort_values(by=["vote_average", "vote_count", "popularity"], ascending=[False, False, False])
    return genre_movies.head(limit).to_dict("records")

def get_all_genres(df: pd.DataFrame):
    """Get sorted list of unique popular genres in the dataset."""
    popular_genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Drama", "Fantasy", "Horror", "Mystery", "Romance",
        "Science Fiction", "Thriller"
    ]
    return popular_genres
