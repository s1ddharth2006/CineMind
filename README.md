# 🎬 CineMind — AI-Powered Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TMDB API](https://img.shields.io/badge/TMDB-API%20v3-01D277.svg?logo=the-movie-database&logoColor=white)](https://themoviedb.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CineMind** is a Netflix-inspired, content-based movie recommendation system. It uses traditional NLP (TF-IDF vectorization) and cosine similarity across 4,800+ films to generate real-time, explainable movie recommendations.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [How It Works & Mathematical Foundations](#-how-it-works--mathematical-foundations)
5. [TMDB API Integration & Fallback Strategy](#-tmdb-api-integration--fallback-strategy)
6. [Tech Stack](#-tech-stack)
7. [Directory Structure](#-directory-structure)
8. [Installation & Setup](#-installation--setup)
9. [Configuration & Environment Variables](#-configuration--environment-variables)
10. [Running the Application](#-running-the-application)
11. [Future Improvements](#-future-improvements)

---

## 🌟 Project Overview

Recommendation systems generally fall into two paradigms: **Collaborative Filtering** (based on user behavior/ratings) and **Content-Based Filtering** (based on item attributes).

CineMind implements a **Content-Based Recommendation Engine** that avoids the "cold start" problem for new items. By analyzing plot summaries, genre classifications, keywords, cast, and directors, it maps every movie into a 5,000-dimensional vector space and computes angular distances between films.

The interface features a Netflix-inspired dark UI with real-time title search/autocomplete, match badges (e.g. `94% Match`), trending carousels, genre filtering, and a detailed movie view.

---

## ✨ Key Features

- **Working ML recommendation algorithm** — no mock data; all recommendations computed live using scikit-learn TF-IDF vectorization and cosine similarity.
- **Explainable output** — each recommendation shows shared directors, genres, cast, or plot themes behind the match.
- **Netflix-style UI** — dark theme, card hover animations, responsive layout.
- **Fast inference** — similarity data is precomputed and stored in `models/similarity.pkl` (<4MB), avoiding a full 4,800×4,800 matrix recalculation on every query.
- **TMDB integration with fallback** — pulls live posters/backdrops from TMDB API v3; falls back gracefully to a local image if no API key or network is unavailable.
- **Genre explorer** — browse top-rated films across Action, Sci-Fi, Drama, Comedy, Thriller, Horror, Romance, Animation.

---

## 🏗️ System Architecture

```
flowchart TD
    subgraph Offline_Pipeline [1. Offline Data & Model Training Pipeline]
        RawData[TMDB 5000 Movies & Credits] --> Merge[Merge on Movie ID]
        Merge --> FeatureEng[Feature Extraction: Genres, Keywords, Cast, Director]
        FeatureEng --> Clean[NLP Tokenization & Normalization]
        Clean --> Combine["Combined Feature Tags: Overview + 2x(Genres) + Keywords + Cast + 2x(Director)"]
        Combine --> TFIDF["TfidfVectorizer(max_features=5000, ngram_range=(1,2))"]
        TFIDF --> CosSim[Cosine Similarity Matrix]
        CosSim --> Serialize["Serialize Models: models/similarity.pkl + data/movies.csv"]
    end

    subgraph Online_Inference [2. Real-Time Inference Engine]
        Serialize --> Loader[src/data_loader.py]
        Serialize --> Recommender[src/recommender.py]
        Query[User Selects Movie Title / ID] --> IndexLookup[Movie Index Lookup]
        IndexLookup --> Recommender
        Recommender --> Rank[Top-10 Ranker & Match Calibration]
        Rank --> Explainer[Explainability Engine: Shared Features]
    end

    subgraph Presentation [3. Streamlit Cinematic UI]
        TMDB[src/tmdb_api.py] <--> Recommender
        TMDB --> Posters[High-Res Posters & Backdrops]
        Recommender --> Streamlit[app.py: Netflix Dark UI]
        Streamlit --> Hero[Hero Section & Search]
        Streamlit --> RecCards[10 Recommended Movie Cards]
        Streamlit --> Trending[Trending / Genre Carousels]
        Streamlit --> Dossier[Detailed Movie View]
    end
```

---

## 🧮 How It Works & Mathematical Foundations

### 1. NLP Preprocessing & Feature Engineering

- **Plot overview**: tokenized and cleaned of punctuation/symbols.
- **Genres & keywords**: extracted from JSON and lowercased.
- **Cast & crew**: top 4 actors + director extracted; spaces removed from names (`Christopher Nolan` → `ChristopherNolan`) so the vectorizer treats them as single tokens instead of splitting on common first names.
- **Feature weighting**: director and genres are duplicated in the combined feature string to give them more weight during TF-IDF.

$$\text{Tags} = \text{Overview} + 2 \times \text{Genres} + \text{Keywords} + \text{Cast} + 2 \times \text{Director}$$

### 2. TF-IDF Vectorization

Text is converted into numeric vectors using TF-IDF with unigrams and bigrams (`ngram_range=(1,2)`), English stop-word filtering, and a 5,000-term vocabulary cap.

$$\text{TF}(t,d) = \frac{f_{t,d}}{\sum_{t'\in d} f_{t',d}} \qquad \text{IDF}(t,D) = \log\left(\frac{1+N}{1+|\{d\in D : t \in d\}|}\right)+1$$

$$\text{TF-IDF}(t,d,D) = \text{TF}(t,d) \times \text{IDF}(t,D)$$

Vectors are L2-normalized so document length doesn't bias similarity.

### 3. Cosine Similarity

$$\cos(\theta) = \frac{\mathbf{A}\cdot\mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$$

Because vectors are already L2-normalized, this simplifies to a dot product — fast to compute at scale.

### 4. Ranking & Explainability

For a selected film: retrieve precomputed similarity scores, drop the film itself, sort descending, take the top 10. For each match, compute shared genres, shared director, and shared cast, and render it as a short explanation (e.g. `Directed by Christopher Nolan • Shared genres: Mystery, Thriller`).

---

## 🌐 TMDB API Integration & Fallback Strategy

Three-tier asset resolution:

1. If a TMDB API key is present → query TMDB API v3 for live poster/backdrop.
2. If unavailable → use a curated offline CDN image map.
3. If that also fails → render a local fallback poster.

The API key is never hardcoded — read from `os.environ["TMDB_API_KEY"]`, `.streamlit/secrets.toml`, or entered directly in the app sidebar.

---

## 💻 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core language |
| ML | Scikit-Learn | TF-IDF, cosine similarity |
| Data processing | Pandas, NumPy | Cleaning, JSON parsing |
| Web framework | Streamlit | Frontend + state |
| API integration | Requests, TMDB API v3 | Live metadata, posters |
| Image generation | Pillow | Fallback poster rendering |

---

## 📁 Directory Structure

```
CineMind/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── movies.csv
├── models/
│   └── similarity.pkl
├── src/
│   ├── recommender.py
│   ├── data_loader.py
│   ├── tmdb_api.py
│   └── preprocessing.py
├── scripts/
│   ├── build_model.py
│   └── create_fallback_poster.py
├── tests/
│   └── test_recommender.py
├── assets/
│   └── fallback_poster.png
└── .streamlit/
    └── config.toml
```

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/s1ddharth2006/CineMind.git
cd CineMind

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ⚙️ Configuration & Environment Variables

CineMind works out of the box in offline fallback mode. To enable live TMDB queries:

1. Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api).
2. Set it as an environment variable:

```bash
# Windows (PowerShell)
$env:TMDB_API_KEY="your_api_key_here"

# Windows (CMD)
set TMDB_API_KEY=your_api_key_here

# macOS / Linux
export TMDB_API_KEY="your_api_key_here"
```

Or add it to `.streamlit/secrets.toml`, or paste it into the app's sidebar.

---

## 🏃 Running the Application

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

Run tests:

```bash
python tests/test_recommender.py
```

Retrain the model:

```bash
python scripts/build_model.py
```

---

## 🔮 Future Improvements

- Hybrid recommendations combining content-based similarity with collaborative filtering (matrix factorization/SVD).
- Upgrade TF-IDF to dense transformer embeddings (`sentence-transformers`) for deeper semantic matching.
- Store user viewing history to build personalized profile vectors.
- Interactive weighting between "more like this director" vs. "more like this genre."

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
