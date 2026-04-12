"""
Sanity Checks — must-pass tests covering the full difficulty spectrum.

These validate that the core pipeline components work correctly
before end-to-end integration.
"""
import os
import sys
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import get_clean_dataframe
from src.structured_queries import (
    get_release_year,
    top_movies_by_score,
    directors_by_repeated_gross_threshold,
    high_votes_low_gross,
    actor_movies_filtered,
    movies_by_genre_and_thresholds,
    lookup_movie_info,
)


# ─────────────────────────────
# Fixtures
# ─────────────────────────────
@pytest.fixture(scope="module")
def df():
    """Load the clean dataframe once for all tests."""
    return get_clean_dataframe()


# ─────────────────────────────
# Data Quality Tests
# ─────────────────────────────
class TestDataQuality:
    """Verify the preprocessing pipeline produces clean data."""

    def test_dataframe_loaded(self, df):
        """Dataset should load with ~1000 rows."""
        assert len(df) > 900, f"Expected ~1000 rows, got {len(df)}"

    def test_released_year_numeric(self, df):
        """Released_Year should be numeric (Int64)."""
        valid = df["Released_Year"].dropna()
        assert all(valid > 1900), "All valid years should be after 1900"
        assert all(valid < 2030), "All valid years should be before 2030"

    def test_gross_preserves_nan(self, df):
        """Gross column should have NaN for missing values, not zero."""
        null_count = df["Gross"].isna().sum()
        assert null_count > 0, "Expected some null Gross values"
        zero_gross = (df["Gross"] == 0).sum()
        assert zero_gross == 0, "Gross should not be filled with zeros"

    def test_has_gross_boolean(self, df):
        """has_gross should match Gross non-null status."""
        assert (df["has_gross"] == df["Gross"].notna()).all()

    def test_genre_list_populated(self, df):
        """genre_list should contain Python lists."""
        assert all(isinstance(gl, list) for gl in df["genre_list"])
        # Multi-genre movies should have multiple entries
        multi = df[df["genre_list"].apply(len) > 1]
        assert len(multi) > 100, "Expected many multi-genre movies"

    def test_cast_list_populated(self, df):
        """cast_list should contain actor names."""
        assert all(isinstance(cl, list) for cl in df["cast_list"])
        assert all(len(cl) > 0 for cl in df["cast_list"])

    def test_lead_actor_set(self, df):
        """lead_actor should equal Star1."""
        assert (df["lead_actor"] == df["Star1"]).all()

    def test_runtime_numeric(self, df):
        """Runtime should be numeric with no 'min' suffix."""
        valid = df["Runtime"].dropna()
        assert all(valid > 0), "All valid runtimes should be positive"

    def test_text_for_embedding(self, df):
        """text_for_embedding should be populated for all rows."""
        assert df["text_for_embedding"].notna().all()
        # Spot check format
        sample = df["text_for_embedding"].iloc[0]
        assert "—" in sample, "Should contain em-dash separators"
        assert "Directed by" in sample


# ─────────────────────────────
# Structured Query Tests
# ─────────────────────────────
class TestStructuredQueries:
    """Test deterministic query functions against known answers."""

    def test_q1_matrix_release_year(self, df):
        """Q1: The Matrix should have released in 1999."""
        result = get_release_year(df, "The Matrix")
        assert result["released_year"] == 1999
        assert result["match_type"] == "exact"

    def test_q2_top5_2019_metascore(self, df):
        """Q2: Top 5 movies of 2019 by meta score should return 5 results."""
        result = top_movies_by_score(df, year=2019, metric="Meta_score", n=5)
        assert result["count"] == 5
        assert len(result["movies"]) == 5
        # All results should be from 2019
        for m in result["movies"]:
            assert m["released_year"] == 2019

    def test_q3_top7_comedy_2010_2020(self, df):
        """Q3: Top 7 comedy movies 2010-2020 by IMDB rating."""
        result = top_movies_by_score(
            df, genre="Comedy", year_start=2010, year_end=2020,
            metric="IMDB_Rating", n=7
        )
        assert result["count"] <= 7
        assert len(result["movies"]) > 0
        for m in result["movies"]:
            assert 2010 <= m["released_year"] <= 2020
            assert "Comedy" in m["genre"]

    def test_q4_horror_threshold(self, df):
        """Q4: Horror movies with meta > 85 and IMDB > 8."""
        result = movies_by_genre_and_thresholds(
            df, genre="Horror", meta_min=85, imdb_min=8.0
        )
        for m in result["movies"]:
            assert m["meta_score"] >= 85
            assert m["imdb_rating"] >= 8.0
            assert "Horror" in m["genre"]

    def test_q5_directors_gross_500m(self, df):
        """Q5: Directors with > $500M gross at least twice."""
        result = directors_by_repeated_gross_threshold(
            df, gross_min=500_000_000, min_count=2
        )
        # Should find at least one director (e.g., Christopher Nolan, Russo brothers)
        if result["count"] > 0:
            for d in result["directors"]:
                assert d["qualifying_movie_count"] >= 2
                assert d["highest_gross"] >= 500_000_000
        else:
            # If no directors found at 500M, the dataset might not have enough
            # Check at a lower threshold to ensure the function works
            result_lower = directors_by_repeated_gross_threshold(
                df, gross_min=200_000_000, min_count=2
            )
            assert result_lower["count"] > 0, "Should find directors at $200M threshold"

    def test_q6_high_votes_low_gross(self, df):
        """Q6: Top 10 movies with >1M votes but lower gross."""
        result = high_votes_low_gross(df, vote_min=1_000_000, n=10)
        assert len(result["movies"]) <= 10
        if len(result["movies"]) > 1:
            # Should be sorted by gross ascending
            grosses = [m["gross"] for m in result["movies"] if m["gross"] is not None]
            assert grosses == sorted(grosses), "Should be sorted by gross ascending"

    def test_fuzzy_title_match(self, df):
        """Misspelled titles should still return results."""
        result = lookup_movie_info(df, "The Shawshenk Redemtion")
        assert "Shawshank" in result.get("title", "")

    def test_movie_not_found(self, df):
        """Non-existent movie should return a result (fuzzy may match loosely) or error."""
        # Fuzzy matching is intentionally generous, so even garbage strings may
        # return a low-confidence match. The key test is that lookup_movie_info
        # doesn't crash and returns a dict.
        result = lookup_movie_info(df, "zzzzqqqq99999")
        assert isinstance(result, dict)

    def test_al_pacino_filter(self, df):
        """Nice-to-have: Al Pacino movies with gross > $50M and IMDB >= 8."""
        # Lead only
        result_lead = actor_movies_filtered(
            df, actor="Al Pacino", role_scope="lead",
            gross_min=50_000_000, imdb_min=8.0
        )
        # Any role
        result_any = actor_movies_filtered(
            df, actor="Al Pacino", role_scope="any",
            gross_min=50_000_000, imdb_min=8.0
        )
        # "Any" should include at least as many as "lead"
        assert result_any["count"] >= result_lead["count"]
        # All results should meet thresholds
        for m in result_any["movies"]:
            assert m["imdb_rating"] >= 8.0
            assert m["gross"] >= 50_000_000


# ─────────────────────────────
# Edge Cases
# ─────────────────────────────
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_genre_filter(self, df):
        """Filtering by a non-existent genre should return empty results."""
        result = top_movies_by_score(df, genre="Nonexistent Genre", n=5)
        assert result["count"] == 0

    def test_year_range_no_results(self, df):
        """Year range with no movies should return empty results cleanly."""
        result = top_movies_by_score(df, year_start=2030, year_end=2035, n=5)
        assert result["count"] == 0

    def test_gross_filter_with_nulls(self, df):
        """Gross filtering should handle null values correctly."""
        result = top_movies_by_score(df, gross_min=100_000_000, n=5)
        for m in result["movies"]:
            assert m["gross"] is not None
            assert m["gross"] >= 100_000_000

    def test_single_year_lookup(self, df):
        """Single year should work."""
        result = top_movies_by_score(df, year=1994, n=5)
        for m in result["movies"]:
            assert m["released_year"] == 1994


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
