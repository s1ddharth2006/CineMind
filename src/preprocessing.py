"""
src/preprocessing.py
Feature engineering and NLP text preprocessing module for CineMind.
"""
import re
import ast

def extract_genres(genre_str):
    """Parse JSON string of genres and return clean space-separated string."""
    if not isinstance(genre_str, str) or not genre_str.strip():
        return ""
    try:
        items = ast.literal_eval(genre_str)
        return " ".join([i["name"].replace(" ", "") for i in items if isinstance(i, dict) and "name" in i])
    except Exception:
        return genre_str

def extract_keywords(kw_str):
    """Parse JSON string of keywords and return clean space-separated string."""
    if not isinstance(kw_str, str) or not kw_str.strip():
        return ""
    try:
        items = ast.literal_eval(kw_str)
        return " ".join([i["name"].replace(" ", "") for i in items if isinstance(i, dict) and "name" in i])
    except Exception:
        return kw_str

def extract_cast(cast_str, top_n=4):
    """Parse JSON string of cast members and return top N actor names with spaces removed."""
    if not isinstance(cast_str, str) or not cast_str.strip():
        return ""
    try:
        items = ast.literal_eval(cast_str)
        # Collapse names (e.g. 'Tom Hanks' -> 'TomHanks') so first & last names are treated as single tokens
        names = []
        for i in items[:top_n]:
            if isinstance(i, dict) and "name" in i:
                names.append(i["name"].replace(" ", ""))
        return " ".join(names)
    except Exception:
        return cast_str

def extract_director(crew_str):
    """Parse JSON string of crew and extract the Director's name with spaces removed."""
    if not isinstance(crew_str, str) or not crew_str.strip():
        return ""
    try:
        items = ast.literal_eval(crew_str)
        for i in items:
            if isinstance(i, dict) and i.get("job") == "Director":
                return i.get("name", "").replace(" ", "")
        return ""
    except Exception:
        return crew_str

def clean_text(text):
    """
    Standardize text: lowercases, strips special characters,
    and removes redundant whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_combined_features(row):
    """
    Constructs an enriched metadata string combining overview, genres,
    keywords, cast, and director.
    """
    overview = str(row.get("overview", "") or "")
    genres = str(row.get("genres", "") or "")
    keywords = str(row.get("keywords", "") or "")
    cast = str(row.get("cast", "") or "")
    director = str(row.get("director", "") or "")

    # Director and genres carry high semantic weight in movie similarity
    combined = f"{overview} {genres} {genres} {keywords} {cast} {director} {director}"
    return clean_text(combined)
