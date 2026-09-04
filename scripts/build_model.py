"""
build_model.py
Data processing, feature engineering, and TF-IDF similarity model training for CineMind.
"""
import os
import ast
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_genres(genre_str):
    try:
        items = ast.literal_eval(genre_str)
        return " ".join([i["name"] for i in items if "name" in i])
    except Exception:
        return ""

def extract_keywords(kw_str):
    try:
        items = ast.literal_eval(kw_str)
        return " ".join([i["name"] for i in items if "name" in i])
    except Exception:
        return ""

def extract_cast(cast_str, top_n=4):
    try:
        items = ast.literal_eval(cast_str)
        return " ".join([i["name"] for i in items[:top_n] if "name" in i])
    except Exception:
        return ""

def extract_director(crew_str):
    try:
        items = ast.literal_eval(crew_str)
        for i in items:
            if i.get("job") == "Director":
                return i.get("name", "")
        return ""
    except Exception:
        return ""

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and clean special characters
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("[START] Starting CineMind model build pipeline...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    movies_path = "data_raw_movies.csv"
    credits_path = "data_raw_credits.csv"

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError("Raw movie or credit data files missing.")

    print("[INFO] Loading raw datasets...")
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    print(f"[INFO] Movies loaded: {len(movies_df)}, Credits loaded: {len(credits_df)}")

    # Merge datasets
    merged = movies_df.merge(credits_df, left_on="id", right_on="movie_id", suffixes=("", "_credit"))

    print("[INFO] Engineering features (genres, keywords, cast, director)...")
    clean_df = pd.DataFrame()
    clean_df["movie_id"] = merged["id"]
    clean_df["title"] = merged["title"].fillna("")
    clean_df["overview"] = merged["overview"].fillna("")
    clean_df["genres"] = merged["genres"].apply(extract_genres)
    clean_df["keywords"] = merged["keywords"].apply(extract_keywords)
    clean_df["cast"] = merged["cast"].apply(extract_cast)
    clean_df["director"] = merged["crew"].apply(extract_director)
    clean_df["release_date"] = merged["release_date"].fillna("")
    clean_df["vote_average"] = merged["vote_average"].fillna(0.0).astype(float)
    clean_df["vote_count"] = merged["vote_count"].fillna(0).astype(int)
    clean_df["popularity"] = merged["popularity"].fillna(0.0).astype(float)

    # Drop any records without title or overview
    clean_df = clean_df.dropna(subset=["title"]).reset_index(drop=True)
    # Remove duplicate movie IDs
    clean_df = clean_df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    print(f"[INFO] Total processed movies: {len(clean_df)}")

    # Combine text features
    # Emphasize director and genres by repeating them slightly
    def make_tags(row):
        return (
            f"{row['overview']} "
            f"{row['genres']} {row['genres']} "
            f"{row['keywords']} "
            f"{row['cast']} "
            f"{row['director']} {row['director']}"
        )

    clean_df["tags"] = clean_df.apply(make_tags, axis=1).apply(clean_text)

    # Save processed movies.csv
    movies_output_csv = os.path.join("data", "movies.csv")
    # Exclude raw tags from saved CSV to keep size minimal, but keep core fields
    save_cols = [
        "movie_id", "title", "overview", "genres", "keywords",
        "cast", "director", "release_date", "vote_average", "vote_count", "popularity"
    ]
    clean_df[save_cols].to_csv(movies_output_csv, index=False)
    print(f"[SUCCESS] Saved clean dataset to {movies_output_csv} ({os.path.getsize(movies_output_csv) / 1024 / 1024:.2f} MB)")

    print("[INFO] Fitting TF-IDF Vectorizer (5,000 features, English stop words)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(clean_df["tags"])
    print(f"[INFO] TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("[INFO] Computing Cosine Similarity...")
    sim_matrix = cosine_similarity(tfidf_matrix)

    print("[INFO] Packing top similarities for instant inference...")
    # Precompute top 50 similar movies for each movie to keep inference under 1ms
    top_similarities = {}
    for idx in range(len(clean_df)):
        sim_scores = list(enumerate(sim_matrix[idx]))
        # Sort by similarity descending, skip self (idx)
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51]
        # Store as list of tuples (movie_idx, rounded_score)
        top_similarities[idx] = [(int(i), round(float(s), 4)) for i, s in sorted_scores]

    # Quick lookup indices
    title_to_index = {title.strip().lower(): idx for idx, title in enumerate(clean_df["title"])}
    id_to_index = {int(m_id): idx for idx, m_id in enumerate(clean_df["movie_id"])}

    model_payload = {
        "top_similarities": top_similarities,
        "title_to_index": title_to_index,
        "id_to_index": id_to_index,
        "tfidf_vectorizer": tfidf,
        "vocab_size": len(tfidf.vocabulary_),
        "num_movies": len(clean_df)
    }

    model_output = os.path.join("models", "similarity.pkl")
    with open(model_output, "wb") as f:
        pickle.dump(model_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[SUCCESS] Model successfully built and saved to {model_output} ({os.path.getsize(model_output) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
