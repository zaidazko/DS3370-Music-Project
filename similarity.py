"""
similarity.py
=============
Checkpoint 3 – Task #1: Similarity Measure for Spotify Listening Data

Computes a weighted composite similarity score (0-1) between two users
based on four dimensions:
  1. Artist overlap        (Jaccard similarity on top-N artists)
  2. Genre distribution    (Cosine similarity on genre ms-played vectors)
  3. Daily listening time  (1 - normalised absolute difference)
  4. Temporal patterns     (Cosine similarity on hour-of-day distributions)

Usage
-----
    python similarity.py --user1 data/user1_music.csv --user2 data/user2_music.csv \
                         --genres1 data/user1_genres.csv --genres2 data/user2_genres.csv

Or import and call compute_similarity() directly.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_music(path: str) -> pd.DataFrame:
    """Load and basic-clean a StreamingHistory_music CSV."""
    df = pd.read_csv(path)

    # Standardise column names (handles minor naming differences)
    df.columns = df.columns.str.strip()
    rename = {
        "trackName": "track",
        "artistName": "artist",
        "Track Description": "track",
        "Artist": "artist",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["endTime"] = pd.to_datetime(df["endTime"], dayfirst=False)
    df["date"]   = df["endTime"].dt.date
    df["hour"]   = df["endTime"].dt.hour

    # Duration in minutes
    df["duration_min"] = df["msPlayed"] / 60_000

    # Play / Skip flag (>=30 s = play)
    df["play"]  = (df["msPlayed"] >= 30_000).astype(int)
    df["skip"]  = (df["msPlayed"] <  30_000).astype(int)

    return df


def load_genres(path: str) -> pd.Series:
    """Load StreamingHistory_genres CSV → Series indexed by genre (ms)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # column names: genre, msPlayed
    s = df.set_index("genre")["msPlayed"]
    return s / s.sum()          # normalise to proportions


# ---------------------------------------------------------------------------
# Component similarity functions (each returns a float in [0, 1])
# ---------------------------------------------------------------------------

def artist_overlap(df1: pd.DataFrame, df2: pd.DataFrame, top_n: int = 50) -> float:
    """Jaccard similarity on the top-N artists (by total ms played) for each user."""
    def top_artists(df):
        return set(
            df.groupby("artist")["msPlayed"]
              .sum()
              .nlargest(top_n)
              .index
        )

    a1, a2 = top_artists(df1), top_artists(df2)
    if not a1 and not a2:
        return 1.0
    return len(a1 & a2) / len(a1 | a2)


def genre_cosine(genres1: pd.Series, genres2: pd.Series) -> float:
    """Cosine similarity between two genre-proportion vectors."""
    all_genres = genres1.index.union(genres2.index)
    v1 = genres1.reindex(all_genres, fill_value=0).values
    v2 = genres2.reindex(all_genres, fill_value=0).values
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def daily_listening_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Similarity based on average daily listening time.
    Uses 1 – |avg1 – avg2| / max(avg1, avg2).
    """
    avg1 = df1.groupby("date")["duration_min"].sum().mean()
    avg2 = df2.groupby("date")["duration_min"].sum().mean()
    if max(avg1, avg2) == 0:
        return 1.0
    return 1.0 - abs(avg1 - avg2) / max(avg1, avg2)


def temporal_cosine(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Cosine similarity on normalised hour-of-day listening distributions.
    Captures whether users listen at similar times of day.
    """
    def hour_dist(df):
        counts = df.groupby("hour")["duration_min"].sum()
        full   = counts.reindex(range(24), fill_value=0).values.astype(float)
        total  = full.sum()
        return full / total if total > 0 else full

    h1, h2 = hour_dist(df1), hour_dist(df2)
    denom = np.linalg.norm(h1) * np.linalg.norm(h2)
    if denom == 0:
        return 0.0
    return float(np.dot(h1, h2) / denom)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_similarity(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    genres1: pd.Series,
    genres2: pd.Series,
    weights: dict = None,
    top_n_artists: int = 50,
) -> dict:
    """
    Compute weighted composite similarity between two users.

    Parameters
    ----------
    df1, df2        : loaded music DataFrames (from load_music)
    genres1, genres2: normalised genre Series (from load_genres)
    weights         : dict with keys artist, genre, daily, temporal (must sum to 1)
    top_n_artists   : number of top artists used for Jaccard similarity

    Returns
    -------
    dict with component scores and final weighted score
    """
    if weights is None:
        weights = {
            "artist":   0.30,
            "genre":    0.30,
            "daily":    0.20,
            "temporal": 0.20,
        }

    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

    scores = {
        "artist":   artist_overlap(df1, df2, top_n=top_n_artists),
        "genre":    genre_cosine(genres1, genres2),
        "daily":    daily_listening_similarity(df1, df2),
        "temporal": temporal_cosine(df1, df2),
    }

    composite = sum(scores[k] * weights[k] for k in scores)

    return {
        "component_scores": scores,
        "weights":          weights,
        "composite":        composite,
    }


# ---------------------------------------------------------------------------
# Evaluation / quality analysis
# ---------------------------------------------------------------------------

def evaluate_similarity(result: dict, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Sanity checks and quality metrics for the similarity measure.

    Returns a dict of evaluation results including:
      - self_similarity: should be 1.0 for user1 vs user1
      - weight_sensitivity: composite score for ±0.10 weight perturbations
      - shared_artist_count / total_artist_pool: raw overlap stats
    """
    # 1. Self-similarity sanity check
    self_check    = artist_overlap(df1, df1)      # should be 1.0
    temporal_self = temporal_cosine(df1, df1)     # should be 1.0

    # 2. Raw overlap stats
    top50_u1 = set(df1.groupby("artist")["msPlayed"].sum().nlargest(50).index)
    top50_u2 = set(df2.groupby("artist")["msPlayed"].sum().nlargest(50).index)
    shared   = top50_u1 & top50_u2

    # 3. Weight sensitivity: shift 10% from artist to genre
    alt_weights = {"artist": 0.20, "genre": 0.40, "daily": 0.20, "temporal": 0.20}
    alt_composite = sum(
        result["component_scores"][k] * alt_weights[k]
        for k in result["component_scores"]
    )

    return {
        "self_similarity_artist_jaccard":   self_check,
        "self_similarity_temporal_cosine":  temporal_self,
        "shared_top50_artists":             len(shared),
        "shared_artist_names":              sorted(shared),
        "total_artist_pool":                len(top50_u1 | top50_u2),
        "composite_default_weights":        result["composite"],
        "composite_genre_heavy_weights":    alt_composite,
        "weight_sensitivity_delta":         abs(result["composite"] - alt_composite),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_similarity_radar(result: dict, output_path: str = "similarity_radar.png"):
    """Radar / spider chart of the four component scores."""
    labels  = list(result["component_scores"].keys())
    values  = [result["component_scores"][k] for k in labels]

    angles  = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values  += [values[0]]          # close the polygon
    angles  += [angles[0]]
    labels  += [labels[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#1DB954")
    ax.fill(angles, values, alpha=0.25, color="#1DB954")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(
        f"User Similarity Radar\nComposite Score: {result['composite']:.3f}",
        fontsize=14, pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Radar chart saved → {output_path}")


def plot_genre_comparison(genres1: pd.Series, genres2: pd.Series,
                          label1: str = "User 1", label2: str = "User 2",
                          output_path: str = "genre_comparison.png", top_n: int = 15):
    """Grouped bar chart comparing top genre proportions."""
    all_genres = genres1.index.union(genres2.index)
    combined   = pd.DataFrame({label1: genres1, label2: genres2}, index=all_genres).fillna(0)

    # Show only genres where at least one user has ≥ 1 % share
    combined = combined[(combined[label1] >= 0.01) | (combined[label2] >= 0.01)]
    combined = combined.nlargest(top_n, label1)

    ax = combined.plot(kind="bar", figsize=(14, 6), color=["#1DB954", "#FF4500"])
    ax.set_title("Genre Distribution Comparison (Top Genres)", fontsize=14)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Proportion of Listening Time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Genre comparison chart saved → {output_path}")


def plot_temporal_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                             label1: str = "User 1", label2: str = "User 2",
                             output_path: str = "temporal_comparison.png"):
    """Line chart of normalised hourly listening distributions."""
    def hour_dist(df):
        counts = df.groupby("hour")["duration_min"].sum()
        full   = counts.reindex(range(24), fill_value=0).values.astype(float)
        return full / full.sum() if full.sum() > 0 else full

    h1, h2 = hour_dist(df1), hour_dist(df2)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(24), h1, "o-", label=label1, color="#1DB954", linewidth=2)
    ax.plot(range(24), h2, "s--", label=label2, color="#FF4500", linewidth=2)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.set_title("Hourly Listening Pattern Comparison", fontsize=14)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Proportion of Listening Time")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Temporal comparison chart saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute Spotify listening similarity between two users")
    parser.add_argument("--user1",   required=True, help="Path to user 1 StreamingHistory_music.csv")
    parser.add_argument("--user2",   required=True, help="Path to user 2 StreamingHistory_music.csv")
    parser.add_argument("--genres1", required=True, help="Path to user 1 StreamingHistory_genres.csv")
    parser.add_argument("--genres2", required=True, help="Path to user 2 StreamingHistory_genres.csv")
    parser.add_argument("--label1",  default="User 1", help="Label for user 1")
    parser.add_argument("--label2",  default="User 2", help="Label for user 2")
    parser.add_argument("--top_n",   type=int, default=50, help="Top-N artists for Jaccard similarity")
    parser.add_argument("--outdir",  default=".", help="Output directory for charts")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df1      = load_music(args.user1)
    df2      = load_music(args.user2)
    genres1  = load_genres(args.genres1)
    genres2  = load_genres(args.genres2)

    print("Computing similarity...")
    result   = compute_similarity(df1, df2, genres1, genres2, top_n_artists=args.top_n)
    eval_out = evaluate_similarity(result, df1, df2)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SIMILARITY REPORT")
    print("="*55)
    print(f"\n  {'COMPONENT':<22}  {'SCORE':>7}  {'WEIGHT':>7}  {'CONTRIBUTION':>12}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*12}")
    for k, score in result["component_scores"].items():
        w   = result["weights"][k]
        con = score * w
        print(f"  {k.capitalize():<22}  {score:>7.4f}  {w:>7.2f}  {con:>12.4f}")
    print(f"\n  {'COMPOSITE SCORE':<22}  {result['composite']:>7.4f}")
    print()
    print(f"  Interpretation:")
    c = result["composite"]
    if   c >= 0.75: label = "Very similar listeners"
    elif c >= 0.50: label = "Moderately similar listeners"
    elif c >= 0.25: label = "Some overlap, mostly different"
    else:           label = "Very different listeners"
    print(f"  → {c:.3f}  ({label})")

    print("\n" + "="*55)
    print("  QUALITY EVALUATION")
    print("="*55)
    print(f"  Self-similarity (artist Jaccard):  {eval_out['self_similarity_artist_jaccard']:.4f}  (expect 1.0)")
    print(f"  Self-similarity (temporal cosine): {eval_out['self_similarity_temporal_cosine']:.4f}  (expect 1.0)")
    print(f"  Shared top-50 artists:             {eval_out['shared_top50_artists']} / {eval_out['total_artist_pool']}")
    if eval_out["shared_artist_names"]:
        print(f"  Shared artists: {', '.join(eval_out['shared_artist_names'][:10])}")
    print(f"  Composite (default weights):       {eval_out['composite_default_weights']:.4f}")
    print(f"  Composite (genre-heavy weights):   {eval_out['composite_genre_heavy_weights']:.4f}")
    print(f"  Weight sensitivity Δ:              {eval_out['weight_sensitivity_delta']:.4f}  (lower = more stable)")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_similarity_radar(result,         output_path=str(out / "similarity_radar.png"))
    plot_genre_comparison(genres1, genres2,
                          label1=args.label1, label2=args.label2,
                          output_path=str(out / "genre_comparison.png"))
    plot_temporal_comparison(df1, df2,
                              label1=args.label1, label2=args.label2,
                              output_path=str(out / "temporal_comparison.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
