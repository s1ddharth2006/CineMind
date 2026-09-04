"""
test_recommender.py
Unit test for the CineMind ContentRecommender engine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.recommender import ContentRecommender

def test_engine():
    rec = ContentRecommender()
    assert rec.is_ready, "Recommender should be initialized"
    assert rec.num_movies > 4000, "Should have loaded over 4000 movies"

    test_movies = ["Inception", "Interstellar", "The Dark Knight", "Avatar", "Titanic"]

    for movie in test_movies:
        recommendations = rec.get_recommendations(movie, top_n=10)
        assert len(recommendations) == 10, f"Expected 10 recommendations for {movie}, got {len(recommendations)}"
        print(f"\n==========================================")
        print(f"Movies Similar to: {movie}")
        print(f"==========================================")
        for i, r in enumerate(recommendations, 1):
            title = r["title"]
            year = r["year"]
            sim = r["similarity_score"]
            pct = r["match_percentage"]
            summary = r["explanation"]["summary"]
            print(f"{i:2d}. {title:<30} ({year}) | Score: {sim:.3f} ({pct}% Match) | {summary}")

    # Test unknown movie handling
    unknown = rec.get_recommendations("A Completely Nonexistent Movie 99999", top_n=10)
    assert unknown == [], "Unknown movie should return empty list without crashing"
    print("\n[SUCCESS] Unknown movie gracefully handled without crashing.")

    # Test movie ID lookup
    by_id = rec.get_recommendations(27205, top_n=5) # Inception ID
    assert len(by_id) == 5, "Should return recommendations by movie ID"
    print("[SUCCESS] Movie ID recommendations verified.")

if __name__ == "__main__":
    test_engine()
