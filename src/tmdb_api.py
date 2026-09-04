"""
src/tmdb_api.py
TMDB API integration client with caching, error resilience, and graceful fallbacks.
"""
import os
import json
import base64
import time
import requests
from typing import Optional, Dict, Any, List

# Streamlit caching support
try:
    import streamlit as st
    cache_data_decorator = st.cache_data
except ImportError:
    def cache_data_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_W500 = "https://image.tmdb.org/t/p/w500"
TMDB_IMAGE_BASE_W1280 = "https://image.tmdb.org/t/p/w1280"
TMDB_IMAGE_BASE_ORIGINAL = "https://image.tmdb.org/t/p/original"
FALLBACK_POSTER_FILE = "assets/fallback_poster.png"
CACHE_FILE_PATH = "data/poster_cache.json"

# Default TMDB API key fallback (CampusX open API key for educational recommenders)
DEFAULT_TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Curated offline CDN paths for top iconic movies to guarantee rich visuals
CURATED_POSTER_PATHS = {
    19995: "/kyeqWdyUXW608qlYkRqosgPugKR.jpg",   # Avatar
    285: "/jGWvtvqqEkjFrBm74ap7P8vY8Pj.jpg",     # Pirates of the Caribbean
    206647: "/zj8NvoxrgQQ0YDYVdPT3u0R3pYv.jpg",  # Spectre
    49026: "/8c4a8kE7PizaGQQnditMmI1xbRp.jpg",   # The Dark Knight Rises
    49529: "/bD74rG2GzM91kYj2u0gM2rPzK5I.jpg",   # John Carter
    559: "/69ZW7G6At0J2ipGq6ap473bN502.jpg",     # Spider-Man 3
    38757: "/vF1mQcE87z2iUvCgQn3R6r5V0zU.jpg",   # Tangled
    99861: "/7WsyChvgrmaJuHXPIL9o77m5qws.jpg",   # Avengers: Age of Ultron
    767: "/b1L79aLH7Ka2pehy5ze1n9eA49.jpg",      # Harry Potter and the Half-Blood Prince
    27205: "/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",   # Inception
    157336: "/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg",  # Interstellar
    155: "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",     # The Dark Knight
    597: "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",      # Titanic
    24428: "/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",    # The Avengers
    680: "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",      # Pulp Fiction
    550: "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",      # Fight Club
    603: "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",      # The Matrix
    13: "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",       # Forrest Gump
    278: "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",      # The Shawshank Redemption
    238: "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",      # The Godfather
    424: "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",      # Schindler's List
    129: "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",      # Spirited Away
    1578: "/rwsMw9mR4233Uj7iGjPq3u7hFzR.jpg",    # Raging Bull
    98: "/pwpGfTImTG7vSczhdCWOP7NmT9V.jpg",       # Gladiator
    105: "/fNOH9f1aA7XRTzl1sAOxT9pn5HG.jpg",      # Back to the Future
    11: "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",       # Star Wars
    120: "/6oom5QYQ2yQTMJIbnvbkBL9cDK6.jpg",     # The Lord of the Rings: Fellowship
    121: "/rrGlNlzFTrXFNGH2V2qrssRDRiz.jpg",     # The Lord of the Rings: Two Towers
    122: "/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",     # The Lord of the Rings: Return of the King
    807: "/696ewuvWq46J37T3p5E0P7Y997K.jpg",      # Se7en
    1891: "/2l05cFWJacyIsTpsqSgH0wQXe4V.jpg",     # The Empire Strikes Back
    329: "/b1xOxrXG2y2vW76rE8uS3qR11Gv.jpg",      # Jurassic Park
    8587: "/aKxBiOXzyIe969rCsmh963P1N9z.jpg",     # The Lion King
    671: "/wuMc08IPKEatf9rnMNX2izIYYBP.jpg",      # Harry Potter and the Sorcerer's Stone
    10195: "/eHuGQ10kZ4BsrhR0VgnTf4nL3g.jpg",     # Thor
    1771: "/gKzYx79y09q7iH1Y4P8nQ8B8hJz.jpg",     # Captain America
    1726: "/78lPtwv72eTNqFW9COBYI0dWDJa.jpg",     # Iron Man
    299536: "/7WsyChvgrmaJuHXPIL9o77m5qws.jpg",  # Avengers: Infinity War
    299534: "/or06FN3Dka5tukK1e9sl16pB3iy.jpg",  # Avengers: Endgame
    475557: "/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",  # Joker
    372058: "/pjeMs3yqRmFL3giJy4PMXWZTgJJ.jpg",  # Your Name.
    496243: "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",  # Parasite
    # Additional frequently recommended movies:
    10133: "/7wYwmDuCbdKF7sAcLj7eqrNaiam.jpg",   # Cypher
    431: "/upk07d8bT0gW42d25L5VnU9n4aF.jpg",     # Cube
    1124: "/Ag2B2KHKQPukjH7WutmgnnSNurZ.jpg",    # The Prestige
    180: "/qtgFcnwh9dAFLocsDk2ySDVS8UF.jpg",     # Minority Report
    77: "/nzlv62aC0octS5AklAiWpXLX9Z0.jpg",      # Memento
    320: "/riVXh3EimGO0y5dgQxEWPRy5Itg.jpg",     # Insomnia
    59967: "/sNjL6SqErDBE8OUZlrDLkexfsCj.jpg",   # Looper
    62: "/ve72VxNqjGM69Uky4WTo2bK6rfq.jpg",      # 2001: A Space Odyssey
    37686: "/pUWIjaMMYJjeBm5bJyE3mIXdQ62.jpg",   # Super 8
    157: "/8GxhFmfcW81z0F6Ld7e1G0uG6aD.jpg",     # Star Trek III
    187: "/67Lp0Y7s2qU9E0U9M2m5Y9U0uL6.jpg",     # Sin City
}

CURATED_BACKDROP_PATHS = {
    19995: "/vL5LR6WdxWPjAZFRbtA2LO8e7wQ.jpg",   # Avatar
    27205: "/8ZTVqvKDQ8emSGUEMjsS4yHAwrp.jpg",   # Inception
    157336: "/xJHokMbljvjADYdit5fK5VQsXEG.jpg",  # Interstellar
    155: "/dqK9Hag1054tghRQSqLSfrkvQnA.jpg",     # The Dark Knight
    597: "/yDI6D5zq3bgfl24q7a0j4r8f4qT.jpg",      # Titanic
    24428: "/9BBTo63ANSmhC4e6r62OJFuK2GL.jpg",   # Avengers
    550: "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg",      # Fight Club
    603: "/fNG7i7rqMErkcqhohV2a6JWdlKo.jpg",      # The Matrix
    98: "/Ar7kOclw4YjHjA0m16T67mX5uPq.jpg",       # Gladiator
}

def load_fallback_poster_b64() -> str:
    """Reads assets/fallback_poster.png and converts to an inline data URI."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    target_path = os.path.join(base_dir, FALLBACK_POSTER_FILE)
    if os.path.exists(target_path):
        try:
            with open(target_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except Exception:
            pass
    # Minimal 1x1 dark transparent fallback
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

FALLBACK_DATA_URI = load_fallback_poster_b64()

class TMDBClient:
    """Resilient TMDB API client with persistent disk caching and graceful fallbacks."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or self._discover_api_key()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CineMind-Recommender/1.0"})
        self.fallback_data_uri = FALLBACK_DATA_URI
        self._disk_cache: Dict[str, Dict[str, Any]] = {}
        self._trending_cache: List[Dict[str, Any]] = []
        self._trending_cache_time: float = 0.0
        self._load_disk_cache()

    def _discover_api_key(self) -> str:
        """Discovers TMDB API key from secrets, environment, or default fallback."""
        # 1. Environment variable
        env_key = os.environ.get("TMDB_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        # 2. Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
                key = str(st.secrets["TMDB_API_KEY"]).strip()
                if key:
                    return key
        except Exception:
            pass

        # 3. Direct read of .streamlit/secrets.toml
        base_dir = os.path.dirname(os.path.dirname(__file__))
        secrets_file = os.path.join(base_dir, ".streamlit", "secrets.toml")
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TMDB_API_KEY") and "=" in line:
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

        # 4. Safe default key for seamless portfolio experience
        return DEFAULT_TMDB_API_KEY

    def _load_disk_cache(self):
        """Loads cached poster paths from disk."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        resolved_path = os.path.join(base_dir, CACHE_FILE_PATH)
        if os.path.exists(resolved_path):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    self._disk_cache = json.load(f)
            except Exception:
                self._disk_cache = {}
        else:
            self._disk_cache = {}

    def _save_disk_cache(self):
        """Persists updated poster paths to disk."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        resolved_path = os.path.join(base_dir, CACHE_FILE_PATH)
        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                json.dump(self._disk_cache, f)
        except Exception:
            pass

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key and len(self._api_key) > 8)

    def set_api_key(self, key: str):
        """Allows dynamically updating the API key at runtime."""
        if key and isinstance(key, str):
            self._api_key = key.strip()

    def get_poster_url(self, movie_id: int, poster_path: Optional[str] = None) -> str:
        """
        Returns full poster image URL for a movie.
        Resolves via:
        1. Passed poster_path
        2. Local disk cache
        3. Curated offline CDN mappings
        4. TMDB API lookup
        5. Base64 Data URI fallback
        """
        movie_id = int(movie_id) if movie_id else 0
        sid = str(movie_id)

        # 1. If explicit poster path is given and valid
        if poster_path and isinstance(poster_path, str) and poster_path.startswith("/"):
            return f"{TMDB_IMAGE_BASE_W500}{poster_path}"

        # 2. Check disk cache
        if sid in self._disk_cache and self._disk_cache[sid].get("poster_path"):
            return f"{TMDB_IMAGE_BASE_W500}{self._disk_cache[sid]['poster_path']}"

        # 3. Check curated offline paths
        if movie_id in CURATED_POSTER_PATHS:
            return f"{TMDB_IMAGE_BASE_W500}{CURATED_POSTER_PATHS[movie_id]}"

        # 4. If API key is available, query TMDB
        if self.has_api_key and movie_id > 0:
            details = self.get_movie_details(movie_id)
            if details and details.get("poster_path"):
                return f"{TMDB_IMAGE_BASE_W500}{details['poster_path']}"

        # 5. Base64 fallback (never 404s in browser)
        return self.fallback_data_uri

    def get_backdrop_url(self, movie_id: int, backdrop_path: Optional[str] = None) -> Optional[str]:
        """Returns high-resolution backdrop image URL or None if unavailable."""
        movie_id = int(movie_id) if movie_id else 0
        sid = str(movie_id)

        if backdrop_path and isinstance(backdrop_path, str) and backdrop_path.startswith("/"):
            return f"{TMDB_IMAGE_BASE_W1280}{backdrop_path}"

        if sid in self._disk_cache and self._disk_cache[sid].get("backdrop_path"):
            return f"{TMDB_IMAGE_BASE_W1280}{self._disk_cache[sid]['backdrop_path']}"

        if movie_id in CURATED_BACKDROP_PATHS:
            return f"{TMDB_IMAGE_BASE_W1280}{CURATED_BACKDROP_PATHS[movie_id]}"

        if self.has_api_key and movie_id > 0:
            details = self.get_movie_details(movie_id)
            if details and details.get("backdrop_path"):
                return f"{TMDB_IMAGE_BASE_W1280}{details['backdrop_path']}"

        return None

    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Queries TMDB v3 `/movie/{id}` endpoint with local caching.
        """
        if not movie_id:
            return None

        sid = str(movie_id)
        if sid in self._disk_cache and "runtime" in self._disk_cache[sid]:
            return self._disk_cache[sid]

        if not self.has_api_key:
            return None

        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": self._api_key, "language": "en-US"}

        try:
            resp = self.session.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                record = {
                    "poster_path": data.get("poster_path"),
                    "backdrop_path": data.get("backdrop_path"),
                    "runtime": data.get("runtime"),
                    "tagline": data.get("tagline"),
                    "budget": data.get("budget"),
                    "revenue": data.get("revenue"),
                    "vote_average": data.get("vote_average"),
                    "vote_count": data.get("vote_count"),
                    "overview": data.get("overview")
                }
                self._disk_cache[sid] = record
                self._save_disk_cache()
                return record
            elif resp.status_code == 429:
                time.sleep(0.3)
                return None
            else:
                return None
        except Exception:
            return None

    def get_trending_movies(self, time_window: str = "week") -> List[Dict[str, Any]]:
        """Queries TMDB v3 `/trending/movie/{time_window}` endpoint with 1-hour in-memory caching."""
        now = time.time()
        if self._trending_cache and (now - self._trending_cache_time < 3600):
            return self._trending_cache

        if not self.has_api_key:
            return []

        url = f"{TMDB_BASE_URL}/trending/movie/{time_window}"
        params = {"api_key": self._api_key}

        try:
            resp = self.session.get(url, params=params, timeout=3.0)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                formatted = []
                for item in results[:12]:
                    formatted.append({
                        "movie_id": item.get("id"),
                        "title": item.get("title", item.get("original_title", "")),
                        "overview": item.get("overview", ""),
                        "vote_average": round(float(item.get("vote_average", 0.0)), 1),
                        "year": str(item.get("release_date", ""))[:4],
                        "poster_path": item.get("poster_path"),
                        "backdrop_path": item.get("backdrop_path")
                    })
                self._trending_cache = formatted
                self._trending_cache_time = now
                return formatted
        except Exception:
            return self._trending_cache or []

        return self._trending_cache or []

    def get_movie_credits(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Queries TMDB `/movie/{id}/credits` for live cast & crew."""
        if not self.has_api_key or not movie_id:
            return None

        url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits"
        params = {"api_key": self._api_key}

        try:
            resp = self.session.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                cast = [c.get("name") for c in data.get("cast", [])[:5] if c.get("name")]
                director = None
                for crew in data.get("crew", []):
                    if crew.get("job") == "Director":
                        director = crew.get("name")
                        break
                return {"cast": cast, "director": director}
        except Exception:
            return None
        return None

# Global client singleton
_tmdb_client_instance = None

def get_tmdb_client() -> TMDBClient:
    """Returns singleton instance of TMDBClient."""
    global _tmdb_client_instance
    if _tmdb_client_instance is None:
        _tmdb_client_instance = TMDBClient()
    return _tmdb_client_instance
