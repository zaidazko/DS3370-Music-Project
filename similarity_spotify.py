"""
similarity_spotify.py
=====================
Checkpoint 3 – Task #1: Similarity Measure for two Spotify users.

Four similarity dimensions (each 0–1):
  1. artist_jaccard   – Jaccard on top-N artists by ms played
  2. skip_rate_sim    – 1 – |skip_rate_1 – skip_rate_2|
  3. daily_sim        – 1 – normalised difference in avg daily minutes
  4. temporal_cosine  – cosine on normalised hour-of-day distributions

Usage
-----
    python similarity_spotify.py \
        --user1  data/Cleaned_StreamingHistory__1_.csv \
        --user2  data/Cleaned_StreamingHistory.csv \
        --label1 "Zaid" --label2 "User 2" \
        --outdir results/
"""

import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── Loader ────────────────────────────────────────────────────────────────────

def load_spotify(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Artist"]           = df["Artist"].str.strip("'\"")
    df["Track Description"]= df["Track Description"].str.strip("'\"")
    df["endTime"]          = pd.to_datetime(df["endTime"])
    df["Date Played"]      = pd.to_datetime(df["Date Played"])
    df["Duration_Minutes"] = pd.to_numeric(df["Duration_Minutes"], errors="coerce").fillna(0)
    df["Play Count"]       = pd.to_numeric(df["Play Count"],  errors="coerce").fillna(0)
    df["Skip Count"]       = pd.to_numeric(df["Skip Count"],  errors="coerce").fillna(0)
    df["hour"]             = df["endTime"].dt.hour
    return df


# ── Component functions ───────────────────────────────────────────────────────

def artist_jaccard(df1: pd.DataFrame, df2: pd.DataFrame, top_n: int = 50) -> tuple:
    top1 = set(df1.groupby("Artist")["msPlayed"].sum().nlargest(top_n).index)
    top2 = set(df2.groupby("Artist")["msPlayed"].sum().nlargest(top_n).index)
    shared = top1 & top2
    score  = len(shared) / len(top1 | top2) if (top1 | top2) else 1.0
    return score, sorted(shared)


def skip_rate_sim(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    def skip_rate(df):
        total = df["Play Count"].sum() + df["Skip Count"].sum()
        return df["Skip Count"].sum() / total if total > 0 else 0.0
    sr1, sr2 = skip_rate(df1), skip_rate(df2)
    return 1.0 - abs(sr1 - sr2), sr1, sr2


def daily_sim(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    avg1 = df1.groupby("Date Played")["Duration_Minutes"].sum().mean()
    avg2 = df2.groupby("Date Played")["Duration_Minutes"].sum().mean()
    score = 1.0 - abs(avg1 - avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1.0
    return float(score), float(avg1), float(avg2)


def temporal_cosine(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    def hdist(df):
        arr = df.groupby("hour")["Duration_Minutes"].sum().reindex(range(24), fill_value=0).values.astype(float)
        return arr / arr.sum() if arr.sum() > 0 else arr
    h1, h2 = hdist(df1), hdist(df2)
    d = np.linalg.norm(h1) * np.linalg.norm(h2)
    return float(np.dot(h1, h2) / d) if d > 0 else 0.0


# ── Master ────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {"artist": 0.35, "skip_rate": 0.20, "daily": 0.20, "temporal": 0.25}

def compute_similarity(df1, df2, weights=None, top_n=50):
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

    aj, shared         = artist_jaccard(df1, df2, top_n)
    skr, sr1, sr2      = skip_rate_sim(df1, df2)
    ds, avg1, avg2     = daily_sim(df1, df2)
    tc                 = temporal_cosine(df1, df2)

    scores = {"artist": aj, "skip_rate": skr, "daily": ds, "temporal": tc}
    composite = sum(scores[k] * weights[k] for k in scores)

    return dict(
        component_scores = scores,
        weights          = weights,
        composite        = composite,
        shared_artists   = shared,
        n_shared         = len(shared),
        skip_rate_1      = sr1,
        skip_rate_2      = sr2,
        avg_daily_1      = avg1,
        avg_daily_2      = avg2,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(result):
    s = result["component_scores"]
    alt = {
        "artist_heavy":   {"artist":0.55,"skip_rate":0.10,"daily":0.15,"temporal":0.20},
        "temporal_heavy": {"artist":0.25,"skip_rate":0.10,"daily":0.20,"temporal":0.45},
        "equal":          {"artist":0.25,"skip_rate":0.25,"daily":0.25,"temporal":0.25},
        "no_skip":        {"artist":0.45,"skip_rate":0.00,"daily":0.25,"temporal":0.30},
    }
    sensitivity = {n: round(sum(s[k]*w[k] for k in s), 4) for n, w in alt.items()}
    all_c = list(sensitivity.values()) + [result["composite"]]
    stability = round(max(all_c) - min(all_c), 4)

    def interp(score, lo, mid, hi):
        return hi if score >= 0.66 else (mid if score >= 0.33 else lo)

    interpretations = {
        "artist":    f"Artist Jaccard  {s['artist']:.4f}  → " + interp(s["artist"],
                         "No shared top artists", "A few common artists", "Very similar artist taste"),
        "skip_rate": f"Skip-rate sim   {s['skip_rate']:.4f}  → " + interp(s["skip_rate"],
                         "Very different listening patience", "Moderate skip-rate difference", "Similar listening patience"),
        "daily":     f"Daily volume    {s['daily']:.4f}  → " + interp(s["daily"],
                         "Very different daily volume", "Moderate volume difference", "Similar daily listening time"),
        "temporal":  f"Temporal cosine {s['temporal']:.4f}  → " + interp(s["temporal"],
                         "Listen at completely different times", "Partial temporal overlap", "Listen at similar times of day"),
    }
    return dict(weight_sensitivity=sensitivity, stability_range=stability,
                interpretations=interpretations)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_radar(result, label1, label2, outpath):
    labels = ["Artist\nOverlap", "Skip Rate\nSimilarity", "Daily\nListening", "Temporal\nPattern"]
    vals   = [result["component_scores"][k] for k in ("artist","skip_rate","daily","temporal")]
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
    v2 = vals + [vals[0]]; a2 = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(a2, v2, "o-", lw=2, color="#1DB954")
    ax.fill(a2, v2, alpha=0.2, color="#1DB954")
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=11)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=8)
    ax.set_title(f"{label1}  vs  {label2}\nComposite Score: {result['composite']:.3f}",
                 fontsize=13, pad=20)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    print(f"Radar → {outpath}")


def plot_top_artists(df1, df2, label1, label2, outpath, top_n=15):
    """Side-by-side top artists by hours listened."""
    def top(df, n):
        s = df.groupby("Artist")["Duration_Minutes"].sum().nlargest(n) / 60
        return s

    t1, t2 = top(df1, top_n), top(df2, top_n)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, t, label, color in zip(axes, [t1, t2], [label1, label2], ["#1DB954","#FF4500"]):
        ax.barh(t.index[::-1], t.values[::-1], color=color)
        ax.set_title(f"{label} – Top {top_n} Artists", fontsize=13)
        ax.set_xlabel("Hours Listened")
        ax.tick_params(axis="y", labelsize=9)
    plt.suptitle("Top Artists Comparison", fontsize=14, y=1.01)
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Top artists → {outpath}")


def plot_temporal(df1, df2, label1, label2, outpath):
    """Hourly listening distribution overlay."""
    def hdist(df):
        arr = df.groupby("hour")["Duration_Minutes"].sum().reindex(range(24), fill_value=0).values.astype(float)
        return arr / arr.sum() if arr.sum() > 0 else arr

    h1, h2 = hdist(df1), hdist(df2)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(range(24), h1, "o-", label=label1, color="#1DB954", lw=2)
    ax.plot(range(24), h2, "s--", label=label2, color="#FF4500", lw=2)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.set_title("Hourly Listening Pattern", fontsize=14)
    ax.set_xlabel("Hour of Day (UTC)"); ax.set_ylabel("Proportion of Listening Time")
    ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    print(f"Temporal → {outpath}")


def plot_daily(df1, df2, label1, label2, outpath):
    """Monthly avg daily listening time."""
    def monthly(df):
        d = df.groupby("Date Played")["Duration_Minutes"].sum().reset_index()
        d["month"] = d["Date Played"].dt.to_period("M")
        return d.groupby("month")["Duration_Minutes"].mean()

    m1, m2 = monthly(df1), monthly(df2)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, m, label, color in zip(axes, [m1, m2], [label1, label2], ["#1DB954","#FF4500"]):
        step = max(1, len(m)//12)
        ax.plot(m.index.astype(str), m.values, color=color, lw=2)
        ax.axhline(m.mean(), color=color, ls=":", alpha=0.6, label=f"Mean: {m.mean():.1f} min")
        ax.fill_between(range(len(m)), m.values, alpha=0.15, color=color)
        ax.set_xticks(range(0, len(m), step))
        ax.set_xticklabels(list(m.index.astype(str))[::step], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{label} – Avg Daily Listening per Month", fontsize=12)
        ax.set_ylabel("Avg Minutes/Day"); ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    print(f"Daily → {outpath}")


def plot_skip_rate(df1, df2, label1, label2, result, outpath):
    """Bar chart of skip rate comparison + breakdown."""
    labels = [label1, label2]
    skip_rates = [result["skip_rate_1"]*100, result["skip_rate_2"]*100]
    play_rates = [100-s for s in skip_rates]

    fig, ax = plt.subplots(figsize=(7,5))
    x = np.arange(2)
    b1 = ax.bar(x, play_rates,  label="Play Rate",  color=["#1DB954","#FF4500"], alpha=0.85)
    b2 = ax.bar(x, skip_rates,  bottom=play_rates, label="Skip Rate",
                color=["#0d6b33","#b33000"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Percentage (%)"); ax.set_ylim(0,110)
    ax.set_title("Play vs Skip Rate Comparison", fontsize=14)
    ax.legend(fontsize=11)
    for i, (pr, sr) in enumerate(zip(play_rates, skip_rates)):
        ax.text(i, pr/2,       f"{pr:.1f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(i, pr + sr/2,  f"{sr:.1f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    print(f"Skip rate → {outpath}")


def plot_sensitivity(result, eval_out, outpath):
    configs = list(eval_out["weight_sensitivity"].keys()) + ["default"]
    scores  = list(eval_out["weight_sensitivity"].values()) + [result["composite"]]
    colors  = ["#1DB954" if c=="default" else "#aaaaaa" for c in configs]
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(configs, scores, color=colors, edgecolor="white")
    ax.set_ylim(0,1); ax.set_title("Weight Sensitivity Analysis", fontsize=13)
    ax.set_ylabel("Composite Score")
    for b, sc in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{sc:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    print(f"Sensitivity → {outpath}")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result, eval_out, label1, label2):
    c = result["composite"]
    interp = ("Very similar listeners"          if c >= 0.75 else
              "Moderately similar"              if c >= 0.50 else
              "Some overlap, mostly different"  if c >= 0.25 else
              "Very different listeners")
    print(f"\n{'='*62}")
    print(f"  SIMILARITY: {label1}  vs  {label2}")
    print(f"{'='*62}")
    print(f"\n  {'COMPONENT':<22} {'SCORE':>7}  {'WEIGHT':>7}  {'CONTRIB':>8}")
    print(f"  {'-'*22} {'-'*7}  {'-'*7}  {'-'*8}")
    for k in ("artist","skip_rate","daily","temporal"):
        sc = result["component_scores"][k]; w = result["weights"][k]
        print(f"  {k.replace('_',' ').capitalize():<22} {sc:>7.4f}  {w:>7.2f}  {sc*w:>8.4f}")
    print(f"\n  COMPOSITE  {c:.4f}  →  {interp}")
    print(f"\n  Skip rates:    {label1} = {result['skip_rate_1']*100:.1f}%  |  {label2} = {result['skip_rate_2']*100:.1f}%")
    print(f"  Avg daily:     {label1} = {result['avg_daily_1']:.1f} min  |  {label2} = {result['avg_daily_2']:.1f} min")
    shared_str = ", ".join(result["shared_artists"][:10]) or "None"
    print(f"  Shared artists ({result['n_shared']}): {shared_str}")
    print(f"\n{'='*62}")
    print("  QUALITY / SENSITIVITY")
    print(f"{'='*62}")
    for nm, sc in eval_out["weight_sensitivity"].items():
        print(f"  {nm:<22} {sc:.4f}")
    print(f"\n  Score range: {eval_out['stability_range']:.4f}  "
          f"({'stable' if eval_out['stability_range']<0.05 else 'weight-sensitive'})")
    print()
    for v in eval_out["interpretations"].values():
        print(f"  • {v}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user1",   required=True)
    p.add_argument("--user2",   required=True)
    p.add_argument("--label1",  default="User 1")
    p.add_argument("--label2",  default="User 2")
    p.add_argument("--top_n",   type=int, default=50)
    p.add_argument("--outdir",  default=".")
    args = p.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df1 = load_spotify(args.user1)
    df2 = load_spotify(args.user2)

    print("Computing similarity...")
    result   = compute_similarity(df1, df2, top_n=args.top_n)
    eval_out = evaluate(result)
    print_report(result, eval_out, args.label1, args.label2)

    plot_radar(result, args.label1, args.label2,         str(out/"similarity_radar.png"))
    plot_top_artists(df1, df2, args.label1, args.label2, str(out/"top_artists.png"))
    plot_temporal(df1, df2, args.label1, args.label2,    str(out/"temporal_pattern.png"))
    plot_daily(df1, df2, args.label1, args.label2,       str(out/"daily_listening.png"))
    plot_skip_rate(df1, df2, args.label1, args.label2, result, str(out/"skip_rate.png"))
    plot_sensitivity(result, eval_out,                   str(out/"weight_sensitivity.png"))
    print("Done.")

if __name__ == "__main__":
    main()
