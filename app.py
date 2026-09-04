"""
app.py - CineMind: AI-Powered Movie Recommendation System
A Netflix-inspired cinematic web application utilizing Content-Based Filtering
with TF-IDF and Cosine Similarity.
"""
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image

from src.data_loader import (
    load_movie_data,
    get_movie_by_title,
    get_movie_by_id,
    search_movies,
    get_popular_movies,
    get_movies_by_genre,
    get_all_genres,
)
from src.recommender import get_recommender
from src.tmdb_api import get_tmdb_client

# ==========================================
# 1. PAGE CONFIG & THEME SETUP
# ==========================================
st.set_page_config(
    page_title="CineMind | AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CUSTOM CINEMATIC CSS (NETFLIX-INSPIRED)
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;600;700;800;900&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #FFFFFF;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0c 0%, #111115 50%, #08080a 100%);
    }

    /* Top Brand Navigation */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
        padding-top: 0.5rem;
    }
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #E50914 0%, #FF3D47 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }
    .brand-badge {
        background: rgba(229, 9, 20, 0.15);
        color: #FF4D56;
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hero Section */
    .hero-banner {
        position: relative;
        border-radius: 18px;
        padding: 3.5rem 3rem;
        margin-bottom: 2.5rem;
        background: radial-gradient(circle at 80% 20%, rgba(229, 9, 20, 0.22) 0%, rgba(14, 14, 18, 0.95) 75%),
                    linear-gradient(135deg, #181820 0%, #0d0d12 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #E50914, #FF5A62, transparent);
    }
    .hero-headline {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 0.8rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #A0A0B0;
        max-width: 650px;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* Section Headers */
    .section-header-box {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2rem 0 1.2rem 0;
    }
    .section-accent-bar {
        width: 4px;
        height: 24px;
        background: #E50914;
        border-radius: 2px;
    }
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #F5F5F7;
        letter-spacing: -0.3px;
        margin: 0;
    }

    /* Movie Poster Cards */
    .movie-card {
        background: #191922;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        display: flex;
        flex-direction: column;
        height: 100%;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(229, 9, 20, 0.6);
        box-shadow: 0 18px 30px -8px rgba(229, 9, 20, 0.3);
    }
    .poster-img-container {
        position: relative;
        width: 100%;
        padding-top: 150%; /* 2:3 aspect ratio */
        background: #111116;
        overflow: hidden;
    }
    .poster-img-container img {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .movie-card:hover .poster-img-container img {
        transform: scale(1.05);
    }
    .match-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(14, 14, 18, 0.88);
        backdrop-filter: blur(8px);
        color: #46D369;
        font-size: 0.76rem;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid rgba(70, 211, 105, 0.3);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    .card-body {
        padding: 12px 14px 14px 14px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: space-between;
    }
    .card-movie-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 4px;
    }
    .card-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: #8C8C9A;
        margin-bottom: 8px;
    }
    .card-genre-pill {
        font-size: 0.7rem;
        color: #B2B2C0;
        background: rgba(255, 255, 255, 0.06);
        padding: 2px 6px;
        border-radius: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-rating {
        color: #FFC107;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 3px;
    }
    .card-sim-bar-bg {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        margin-top: 6px;
        overflow: hidden;
    }
    .card-sim-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #E50914, #FF5A62);
        border-radius: 2px;
    }

    /* Spotlight / Hero Featured Movie */
    .spotlight-card {
        background: linear-gradient(135deg, rgba(28, 28, 38, 0.8) 0%, rgba(18, 18, 25, 0.95) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 2rem;
        display: flex;
        gap: 28px;
        align-items: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
    }
    @media (max-width: 768px) {
        .spotlight-card {
            flex-direction: column;
            text-align: center;
        }
    }
    .spotlight-poster {
        width: 180px;
        height: 270px;
        border-radius: 12px;
        object-fit: cover;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
        flex-shrink: 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .spotlight-info {
        flex-grow: 1;
    }
    .spotlight-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .spotlight-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        margin-bottom: 14px;
        font-size: 0.9rem;
        color: #A0A0B5;
    }
    .spotlight-overview {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #D0D0DE;
        margin-bottom: 16px;
        max-width: 850px;
    }
    .spotlight-crew {
        font-size: 0.84rem;
        color: #8E8E9E;
        margin-bottom: 6px;
    }
    .spotlight-crew strong {
        color: #BDBDCB;
    }

    /* Buttons Styling */
    div.stButton > button {
        background: #E50914 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.6rem !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.4) !important;
    }
    div.stButton > button:hover {
        background: #F40612 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(229, 9, 20, 0.6) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Chip Buttons */
    .chip-btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
        margin-bottom: 1.5rem;
    }

    /* Modal / Expanded Details */
    .details-panel {
        background: #14141c;
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 16px;
        padding: 24px;
        margin: 1.5rem 0 2rem 0;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.8);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d0d12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    .sidebar-stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .sidebar-stat-num {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #E50914;
    }
    .sidebar-stat-lbl {
        font-size: 0.75rem;
        color: #8C8C9A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Explainability Pill */
    .explain-pill {
        background: rgba(229, 9, 20, 0.1);
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.78rem;
        color: #FF8A90;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 3. INITIALIZATION & SESSION STATE
# ==========================================
if "selected_movie_title" not in st.session_state:
    st.session_state.selected_movie_title = "Inception"
if "details_movie_id" not in st.session_state:
    st.session_state.details_movie_id = None
if "active_genre" not in st.session_state:
    st.session_state.active_genre = "Action"
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = ["Inception"]
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

# Load core modules
@st.cache_resource(show_spinner=False)
def init_system():
    df = load_movie_data("data/movies.csv")
    rec = get_recommender()
    tmdb = get_tmdb_client()
    return df, rec, tmdb

with st.spinner("Initializing CineMind AI Engine..."):
    movies_df, recommender, tmdb_client = init_system()

# ==========================================
# 4. SIDEBAR SETTINGS & ML METRICS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <h1 class="brand-title">CineMind</h1>
            <span class="brand-badge">AI v1.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("AI-Powered Content-Based Movie Recommender")

    st.markdown("---")

    # Architecture & Model Stats
    st.markdown("### 🧠 Engine Specifications")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"""
            <div class="sidebar-stat-card">
                <div class="sidebar-stat-num">{recommender.num_movies:,}</div>
                <div class="sidebar-stat-lbl">Movies Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"""
            <div class="sidebar-stat-card">
                <div class="sidebar-stat-num">{recommender.vocab_size:,}</div>
                <div class="sidebar-stat-lbl">TF-IDF Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="sidebar-stat-card">
            <div style="font-size: 0.85rem; color: #FFFFFF; font-weight: 600;">Algorithm Pipeline</div>
            <div style="font-size: 0.78rem; color: #A0A0B0; margin-top: 4px;">
                • TF-IDF N-Gram Vectorization<br>
                • Cosine Similarity Metric<br>
                • Feature-Weighted Explainability<br>
                • Sub-millisecond Precomputed Matrix
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # TMDB API Key Section
    st.markdown("### 🔑 TMDB Integration")
    api_status_color = "#46D369" if tmdb_client.has_api_key else "#FFA000"
    api_status_text = "Connected (Live TMDB API)" if tmdb_client.has_api_key else "Offline Mode (CDN Fallback)"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {api_status_color};"></div>
            <span style="font-size: 0.85rem; color: #D0D0DE; font-weight: 500;">{api_status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Configure TMDB API Key"):
        st.caption("Add your free TMDB API key from themoviedb.org to enable live trending movies and full high-res posters.")
        custom_key = st.text_input("Enter TMDB API Key", type="password", key="user_tmdb_key")
        if st.button("Save API Key"):
            if custom_key:
                tmdb_client.set_api_key(custom_key)
                st.success("API Key activated successfully!")
                st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #666; text-align: center; margin-top: 1rem;">
            Built with Scikit-learn, Pandas & Streamlit<br>
            Portfolio Project • AI/ML Developer
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# 5. AUTO-SCROLL CONTROLLER & HERO SECTION
# ==========================================
st.markdown("<div id='top-anchor'></div>", unsafe_allow_html=True)

if st.session_state.get("scroll_to_top", False):
    st.session_state.scroll_to_top = False
    components.html(
        """
        <script>
            function smoothScrollToTop() {
                try {
                    const p = window.parent || window;
                    const doc = p.document;
                    const scrollables = [
                        doc.querySelector('[data-testid="stMain"]'),
                        doc.querySelector('.main'),
                        doc.querySelector('section.main'),
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.documentElement,
                        doc.body
                    ];
                    scrollables.forEach(el => {
                        if (el && typeof el.scrollTo === 'function') {
                            el.scrollTo({top: 0, left: 0, behavior: 'smooth'});
                        }
                    });
                    if (p && typeof p.scrollTo === 'function') {
                        p.scrollTo({top: 0, left: 0, behavior: 'smooth'});
                    }
                    const topTarget = doc.getElementById('top-anchor');
                    if (topTarget && typeof topTarget.scrollIntoView === 'function') {
                        topTarget.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                } catch (e) {
                    console.error('Scroll error:', e);
                }
            }
            smoothScrollToTop();
            setTimeout(smoothScrollToTop, 50);
            setTimeout(smoothScrollToTop, 150);
            setTimeout(smoothScrollToTop, 350);
        </script>
        """,
        height=0,
        width=0,
    )
    st.markdown(
        """
        <img src="data:image/png;base64,invalid" style="display:none;" onerror="
            try {
                const p = window.parent || window;
                const doc = p.document;
                const m = doc.querySelector('[data-testid=\\'stMain\\']') || doc.querySelector('.main') || doc.querySelector('section.main');
                if (m) m.scrollTo({top: 0, left: 0, behavior: 'smooth'});
                p.scrollTo({top: 0, left: 0, behavior: 'smooth'});
                const a = doc.getElementById('top-anchor');
                if (a) a.scrollIntoView({behavior: 'smooth', block: 'start'});
            } catch(e) {}
        ">
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-headline">Find Your Next Favorite Movie</div>
        <div class="hero-subtitle">
            Discover movies you'll love using content-based machine learning recommendations.
            Powered by TF-IDF vectorization and cosine similarity over 4,800+ films.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search & Autocomplete Interface
search_col, btn_col = st.columns([4, 1])

movie_titles = movies_df["title"].tolist()
current_selection = st.session_state.selected_movie_title
default_index = movie_titles.index(current_selection) if current_selection in movie_titles else 0

with search_col:
    selected_title = st.selectbox(
        "What movie do you like?",
        options=movie_titles,
        index=default_index,
        help="Type or select any movie title to generate similarity recommendations.",
        key="search_selectbox"
    )

with btn_col:
    st.write("") # Spacer for vertical alignment with selectbox
    st.write("")
    get_rec_clicked = st.button("Get Recommendations", use_container_width=True)

if get_rec_clicked and selected_title:
    st.session_state.selected_movie_title = selected_title
    if selected_title not in st.session_state.recommendation_history:
        st.session_state.recommendation_history.append(selected_title)
    st.session_state.details_movie_id = None
    st.rerun()

# Quick Suggestion Chips
st.markdown("<div style='font-size: 0.85rem; color: #8E8E9E; margin-bottom: 4px;'>Popular Suggestions:</div>", unsafe_allow_html=True)
chip_cols = st.columns(6)
suggestions = ["Inception", "Interstellar", "The Dark Knight", "The Avengers", "Titanic", "Avatar"]
for i, sugg in enumerate(suggestions):
    with chip_cols[i]:
        if st.button(f"🎬 {sugg}", key=f"chip_{sugg}", use_container_width=True):
            st.session_state.selected_movie_title = sugg
            st.session_state.details_movie_id = None
            st.rerun()

# ==========================================
# 6. SELECTED MOVIE SPOTLIGHT
# ==========================================
active_movie = get_movie_by_title(movies_df, st.session_state.selected_movie_title)

if active_movie:
    poster_url = tmdb_client.get_poster_url(active_movie["movie_id"])
    backdrop_url = tmdb_client.get_backdrop_url(active_movie["movie_id"])

    st.markdown(
        f"""
        <div class="spotlight-card">
            <img class="spotlight-poster" src="{poster_url}" alt="{active_movie['title']}" onerror="this.onerror=null; this.src='{tmdb_client.fallback_data_uri}'">
            <div class="spotlight-info">
                <div class="spotlight-title">{active_movie['title']}</div>
                <div class="spotlight-meta">
                    <span style="color: #46D369; font-weight: 700;">★ {active_movie['vote_average']} Rating</span>
                    <span>•</span>
                    <span>{active_movie['year']}</span>
                    <span>•</span>
                    <span>{active_movie['genres']}</span>
                </div>
                <div class="spotlight-overview">{active_movie['overview']}</div>
                <div class="spotlight-crew"><strong>Director:</strong> {active_movie['director'] or 'N/A'}</div>
                <div class="spotlight-crew"><strong>Cast:</strong> {active_movie['cast'] or 'N/A'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# 7. MOVIE DETAILS DIALOG (NATIVE & INSTANT)
# ==========================================
@st.dialog("🎬 Movie Dossier", width="large")
def show_movie_details_modal(movie):
    m_id = movie["movie_id"]
    poster_src = tmdb_client.get_poster_url(m_id)
    d_details = tmdb_client.get_movie_details(m_id) or {}
    runtime_val = d_details.get("runtime") or movie.get("runtime")
    runtime_str = f"{runtime_val} min" if runtime_val else "Feature Film"

    col_l, col_r = st.columns([1, 1.8])
    with col_l:
        st.markdown(
            f"""
            <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.7); border: 1px solid rgba(255,255,255,0.12);">
                <img src="{poster_src}" style="width: 100%; height: auto; display: block;" onerror="this.onerror=null; this.src='{tmdb_client.fallback_data_uri}'">
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_r:
        st.markdown(f"## {movie['title']} ({movie.get('year', 'N/A')})")
        st.markdown(f"⭐ **{movie.get('vote_average', '7.0')} / 10** • ⏱️ **{runtime_str}** • 🏷️ *{movie.get('genres', '')}*")

        if "similarity_score" in movie:
            raw_s = movie["similarity_score"]
            st.markdown(f"🎯 **Similarity Score:** `{raw_s:.3f}` • **Algorithm Match:** `{movie.get('match_percentage', 85)}%`")

        st.markdown("### Synopsis")
        st.markdown(f"<div style='color: #D0D0DE; line-height: 1.6; font-size: 0.95rem; margin-bottom: 12px;'>{movie.get('overview', 'No overview available.')}</div>", unsafe_allow_html=True)

        st.markdown(f"**Directed by:** {movie.get('director') or 'N/A'}")
        st.markdown(f"**Starring:** {movie.get('cast') or 'N/A'}")

        if movie.get("explanation", {}).get("summary"):
            st.markdown(f"<div class='explain-pill' style='margin-top: 10px; margin-bottom: 14px;'>💡 {movie['explanation']['summary']}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"🎯 Recommend Movies Similar to {movie['title']}", key=f"dialog_rec_btn_{m_id}", use_container_width=True):
            st.session_state.selected_movie_title = movie['title']
            if movie['title'] not in st.session_state.recommendation_history:
                st.session_state.recommendation_history.append(movie['title'])
            st.session_state.scroll_to_top = True
            st.rerun()

# ==========================================
# 8. RECOMMENDATION RESULTS (TOP 10)
# ==========================================
st.markdown(
    f"""
    <div class="section-header-box">
        <div class="section-accent-bar"></div>
        <div class="section-title">Movies Similar to {st.session_state.selected_movie_title}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

recommendations = recommender.get_recommendations(st.session_state.selected_movie_title, top_n=10)

if not recommendations:
    st.warning(f"No recommendations found for '{st.session_state.selected_movie_title}'. Try searching for another title.")
else:
    # Calibrate display match percentages for realistic presentation (e.g. 98% down to 78%)
    max_raw = max([r["similarity_score"] for r in recommendations]) if recommendations else 1.0
    min_raw = min([r["similarity_score"] for r in recommendations]) if recommendations else 0.0

    # Render in 2 rows of 5 cards
    row1 = recommendations[:5]
    row2 = recommendations[5:10]

    for row_idx, row_movies in enumerate([row1, row2]):
        cols = st.columns(5)
        for col_idx, movie in enumerate(row_movies):
            with cols[col_idx]:
                # Non-linear scaled match percentage (98% down to ~76%)
                raw_s = movie["similarity_score"]
                if max_raw > min_raw:
                    calibrated_pct = int(76 + ((raw_s - min_raw) / (max_raw - min_raw)) * 22)
                else:
                    calibrated_pct = movie["match_percentage"]

                poster_src = tmdb_client.get_poster_url(movie["movie_id"])
                first_genre = movie["genres"].split()[0] if movie["genres"] else "Film"

                card_html = f"""
                <div class="movie-card">
                    <div class="poster-img-container">
                        <img src="{poster_src}" alt="{movie['title']}" onerror="this.onerror=null; this.src='{tmdb_client.fallback_data_uri}'">
                        <div class="match-badge">{calibrated_pct}% Match</div>
                    </div>
                    <div class="card-body">
                        <div>
                            <div class="card-movie-title" title="{movie['title']}">{movie['title']}</div>
                            <div class="card-meta">
                                <span>{movie['year']}</span>
                                <span>•</span>
                                <span class="card-genre-pill">{first_genre}</span>
                                <span>•</span>
                                <span class="card-rating">★ {movie['vote_average']}</span>
                            </div>
                            <div class="card-sim-bar-bg">
                                <div class="card-sim-bar-fill" style="width: {calibrated_pct}%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Action button to view details / pivot recommendation
                card_key = f"card_rec_{movie['movie_id']}_{row_idx}_{col_idx}"
                if st.button("More Info", key=card_key, use_container_width=True):
                    show_movie_details_modal(movie)

                # ML Explainability Pill
                summary_text = movie.get("explanation", {}).get("summary", "")
                if summary_text:
                    st.markdown(f"<div class='explain-pill'>💡 {summary_text}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 9. TRENDING MOVIES ROW
# ==========================================
st.markdown(
    """
    <div class="section-header-box">
        <div class="section-accent-bar"></div>
        <div class="section-title">🔥 Trending Movies</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch cached trending movies (zero latency on reruns)
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_trending():
    trending = tmdb_client.get_trending_movies()
    if not trending:
        trending = get_popular_movies(movies_df, limit=10)
    return trending

trending_movies = get_cached_trending()

t_cols = st.columns(5)
for idx, t_movie in enumerate(trending_movies[:5]):
    with t_cols[idx]:
        p_src = tmdb_client.get_poster_url(t_movie["movie_id"], t_movie.get("poster_path"))
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="poster-img-container">
                    <img src="{p_src}" alt="{t_movie['title']}" onerror="this.onerror=null; this.src='{tmdb_client.fallback_data_uri}'">
                </div>
                <div class="card-body">
                    <div class="card-movie-title" title="{t_movie['title']}">{t_movie['title']}</div>
                    <div class="card-meta">
                        <span>{t_movie.get('year', 'N/A')}</span>
                        <span>•</span>
                        <span class="card-rating">★ {t_movie.get('vote_average', '7.5')}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        t_btn_c1, t_btn_c2 = st.columns(2)
        with t_btn_c1:
            if st.button("Select", key=f"trend_btn_{t_movie['movie_id']}_{idx}", use_container_width=True):
                st.session_state.selected_movie_title = t_movie["title"]
                st.session_state.scroll_to_top = True
                st.rerun()
        with t_btn_c2:
            if st.button("Info", key=f"trend_info_{t_movie['movie_id']}_{idx}", use_container_width=True):
                show_movie_details_modal(t_movie)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 10. POPULAR GENRES EXPLORER
# ==========================================
st.markdown(
    """
    <div class="section-header-box">
        <div class="section-accent-bar"></div>
        <div class="section-title">🎭 Explore by Genre</div>
    </div>
    """,
    unsafe_allow_html=True,
)

popular_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Animation"]
genre_cols = st.columns(len(popular_genres))

for i, g in enumerate(popular_genres):
    with genre_cols[i]:
        # Handle Science Fiction mapping
        lookup_g = "Science Fiction" if g == "Sci-Fi" else g
        is_active = (st.session_state.active_genre == lookup_g)
        btn_label = f"🔥 {g}" if is_active else g
        if st.button(btn_label, key=f"genre_btn_{g}", use_container_width=True):
            st.session_state.active_genre = lookup_g
            st.rerun()

# Display movies from the selected genre
genre_movies = get_movies_by_genre(movies_df, st.session_state.active_genre, limit=5)

g_cols = st.columns(5)
for idx, g_movie in enumerate(genre_movies):
    with g_cols[idx]:
        p_src = tmdb_client.get_poster_url(g_movie["movie_id"])
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="poster-img-container">
                    <img src="{p_src}" alt="{g_movie['title']}" onerror="this.onerror=null; this.src='{tmdb_client.fallback_data_uri}'">
                </div>
                <div class="card-body">
                    <div class="card-movie-title" title="{g_movie['title']}">{g_movie['title']}</div>
                    <div class="card-meta">
                        <span>{g_movie['year']}</span>
                        <span>•</span>
                        <span class="card-rating">★ {g_movie['vote_average']}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        g_btn_c1, g_btn_c2 = st.columns(2)
        with g_btn_c1:
            if st.button("Select", key=f"g_btn_{g_movie['movie_id']}_{idx}", use_container_width=True):
                st.session_state.selected_movie_title = g_movie["title"]
                st.session_state.scroll_to_top = True
                st.rerun()
        with g_btn_c2:
            if st.button("Info", key=f"g_info_{g_movie['movie_id']}_{idx}", use_container_width=True):
                show_movie_details_modal(g_movie)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding: 2rem 0; text-align: center; color: #707080; font-size: 0.85rem;">
        <span style="font-family: 'Outfit'; font-weight: 700; color: #E50914;">CINEMIND</span> — AI-Powered Movie Recommendation System<br>
        Engineered with Python, Scikit-Learn, Pandas, Streamlit & TMDB API.
    </div>
    """,
    unsafe_allow_html=True,
)
