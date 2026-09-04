"""
scripts/verify_posters.py
Verify that all recommendations retrieve valid TMDB poster URLs.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.recommender import ContentRecommender
from src.tmdb_api import get_tmdb_client

def test_posters():
    rec = ContentRecommender()
    tmdb = get_tmdb_client()

    movies_to_check = ["Inception", "The Dark Knight", "Interstellar", "Avatar", "Titanic"]

    all_success = True
    total_checked = 0
    total_tmdb = 0

    for title in movies_to_check:
        print(f"\nChecking posters for recommendations of: {title}")
        recs = rec.get_recommendations(title, top_n=10)
        for r in recs:
            total_checked += 1
            poster_url = tmdb.get_poster_url(r["movie_id"])
            if poster_url.startswith("https://image.tmdb.org"):
                status = "TMDB CDN (SUCCESS)"
                total_tmdb += 1
            elif poster_url.startswith("data:image"):
                status = "BASE64 (CLEAN FALLBACK)"
            else:
                status = "INVALID"
                all_success = False
            print(f"  - {r['title'][:25]:<26} ({r['year']}) : {status} -> {poster_url[:55]}...")

    print(f"\n==========================================")
    print(f"Total posters checked: {total_checked}")
    print(f"Total loaded directly from TMDB CDN: {total_tmdb} ({total_tmdb/total_checked*100:.1f}%)")
    print(f"Verification status: {'PASSED' if all_success else 'FAILED'}")
    print(f"==========================================")

if __name__ == "__main__":
    test_posters()
