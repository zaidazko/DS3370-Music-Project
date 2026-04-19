"""
clustering_spotify.py
=====================
Checkpoint 3 – Task #3: Clustering Spotify Data

Perform clustering using K-Means to see if two people's data are separable,
and explain what the clusters indicate.

Usage
-----
    python clustering_spotify.py \
        --user1 SpotifyData/data/processed/Cleaned_StreamingHistory.csv \
        --user2 SpotifyData2/data/processed/Cleaned_StreamingHistory.csv \
        --label1 "User 1" --label2 "User 2" \
        --outdir results/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ── Loader ────────────────────────────────────────────────────────────────────

def load_and_prepare_data(path1: str, path2: str, label1: str, label2: str) -> pd.DataFrame:
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    print(f"  {label1}: {len(df1):,} rows  |  {label2}: {len(df2):,} rows")

    # Subsample the larger dataset so K-Means isn't dominated by one user
    min_size = min(len(df1), len(df2))
    if len(df1) > min_size:
        df1 = df1.sample(n=min_size, random_state=42)
    if len(df2) > min_size:
        df2 = df2.sample(n=min_size, random_state=42)
    print(f"  After balancing: {len(df1):,} rows each")

    # Assign ground truth labels
    df1['true_user'] = 0
    df1['user_label'] = label1

    df2['true_user'] = 1
    df2['user_label'] = label2

    # Combine datasets
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Feature Engineering
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['hour'] = df['endTime'].dt.hour
    df['dayofweek'] = df['endTime'].dt.dayofweek
    df['month'] = df['endTime'].dt.month
    
    # Build skip indicator from whichever column is available per row.
    # After concat, User 2 has 'skipped' while User 1 only has 'Skip Count',
    # so we must combine both sources rather than treating them as exclusive.
    df['skipped_int'] = 0
    if 'skipped' in df.columns:
        skipped_from_bool = (
            df['skipped'].astype(str).str.lower()
            .map({'true': 1, 'false': 0})
        )
        mask_bool = skipped_from_bool.notna()
        df.loc[mask_bool, 'skipped_int'] = skipped_from_bool[mask_bool].astype(int)
    if 'Skip Count' in df.columns:
        mask_skip_count = (df['skipped_int'] == 0) & df['Skip Count'].notna() & (df['Skip Count'] > 0)
        df.loc[mask_skip_count, 'skipped_int'] = 1
        
    # Ensure Duration is numeric and fill NaNs
    df['Duration_Minutes'] = pd.to_numeric(df['Duration_Minutes'], errors='coerce').fillna(0)
    
    return df


# ── Clustering ────────────────────────────────────────────────────────────────

def perform_clustering(df: pd.DataFrame, features: list):
    # Select features for clustering
    X = df[features].copy()
    
    # Drop rows with NaNs in the selected features
    valid_idx = X.dropna().index
    X_valid = X.loc[valid_idx]
    
    # Standardize features (essential for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # Perform KMeans clustering (k=2 for the two users)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels back to the dataframe
    df.loc[valid_idx, 'cluster'] = cluster_labels
    
    # Get cluster centers in original scale
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    
    return df, centers, kmeans, scaler, valid_idx


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cluster_composition(df, valid_idx, label1, label2, outpath):
    """Stacked Bar chart showing the percentage of each user in each cluster."""
    plot_df = df.loc[valid_idx]
    
    counts = plot_df.groupby(['cluster', 'user_label']).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    percentages.plot(kind='bar', stacked=True, ax=ax, color=["#1DB954", "#FF4500"], alpha=0.85)
    
    ax.set_title("Cluster Composition by User\n(Are they separable?)", fontsize=14)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Percentage of Sessions (%)", fontsize=12)
    ax.set_xticklabels(["Cluster 0", "Cluster 1"], rotation=0)
    ax.set_ylim(0, 100)
    
    # Place legend outside
    ax.legend(title="User", fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add text annotations for percentages
    for n, x in enumerate([*percentages.index.values]):
        for (proportion, y_loc) in zip(percentages.loc[x], percentages.loc[x].cumsum()):
            # Only add text if the proportion is large enough to fit
            if proportion > 5:
                ax.text(x=n,
                        y=(y_loc - proportion) + (proportion / 2),
                        s=f'{np.round(proportion, 1)}%', 
                        color="white",
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center")
                    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Composition Plot → {outpath}")


def plot_cluster_profiles(centers, outpath):
    """Bar chart showing the average feature values for each cluster."""
    # Normalize centers just for visualization comparison (min-max scaling per feature)
    normalized_centers = (centers - centers.min()) / (centers.max() - centers.min() + 1e-6)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(centers.columns))
    width = 0.35
    
    ax.bar(x - width/2, normalized_centers.iloc[0], width, label='Cluster 0', color="#3498db")
    ax.bar(x + width/2, normalized_centers.iloc[1], width, label='Cluster 1', color="#e74c3c")
    
    ax.set_ylabel('Relative Value (Min-Max Scaled)')
    ax.set_title('What do the clusters indicate? (Cluster Profiles)')
    ax.set_xticks(x)
    ax.set_xticklabels(centers.columns, fontsize=11)
    ax.legend()
    
    # Add the actual values as text
    for i in range(len(centers.columns)):
        val0 = centers.iloc[0, i]
        val1 = centers.iloc[1, i]
        ax.text(i - width/2, normalized_centers.iloc[0, i] + 0.02, f"{val0:.2f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, normalized_centers.iloc[1, i] + 0.02, f"{val1:.2f}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Profiles Plot    → {outpath}")


def plot_scatter_2d(df, centers, valid_idx, outpath):
    """Scatter plot of two specific features with cluster centroids."""
    plot_df = df.loc[valid_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data points, colored by cluster
    sns.scatterplot(
        data=plot_df, 
        x='hour', 
        y='Duration_Minutes', 
        hue='cluster', 
        palette=["#3498db", "#e74c3c"],
        alpha=0.05,  # Highly transparent due to many points
        s=15,        # Small point size
        edgecolor=None,
        ax=ax
    )
    
    # Plot the centroids
    ax.scatter(
        centers['hour'], 
        centers['Duration_Minutes'], 
        color='black', 
        marker='X', 
        s=300, 
        linewidths=3, 
        label='Centroids',
        zorder=5
    )
    
    ax.set_title("Cluster Scatter Plot (Hour vs Duration)", fontsize=14)
    # Limit y-axis so massive outliers don't squish the plot
    ax.set_ylim(0, plot_df['Duration_Minutes'].quantile(0.99)) 
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"2D Scatter Plot  → {outpath}")


def plot_pca_scatter(df, valid_idx, kmeans_model, scaler_model, features, outpath):
    """PCA scatter plot reducing all features to 2D with centroids."""
    from sklearn.decomposition import PCA
    
    plot_df = df.loc[valid_idx]
    X_scaled = scaler_model.transform(plot_df[features])
    
    # Reduce from 4 dimensions down to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Also transform the centroids into this 2D space
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the PCA points
    scatter = ax.scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        c=plot_df['cluster'], 
        cmap=matplotlib.colors.ListedColormap(["#3498db", "#e74c3c"]),
        alpha=0.05, 
        s=10,
        edgecolors='none'
    )
    
    # Create custom legend for the clusters
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Cluster 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Cluster 1'),
        Line2D([0], [0], marker='X', color='w', markeredgecolor='black', markerfacecolor='black', markersize=12, label='Centroids')
    ]
    
    # Plot the PCA centroids
    ax.scatter(
        centroids_pca[:, 0], 
        centroids_pca[:, 1], 
        color='black', 
        marker='X', 
        s=300, 
        linewidths=3,
        zorder=5
    )
    
    ax.set_title("PCA Scatter Plot of Clusters (All Features Combined)", fontsize=14)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"PCA Scatter Plot → {outpath}")


def plot_feature_distributions(df, valid_idx, outpath):
    """Violin plots of feature distributions separated by cluster."""
    plot_df = df.loc[valid_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    features_to_plot = ['hour', 'dayofweek', 'Duration_Minutes', 'skipped_int']
    titles = ['Hour of Day', 'Day of Week', 'Duration (Minutes)', 'Skip Rate']
    
    for i, (feature, title) in enumerate(zip(features_to_plot, titles)):
        if feature == 'skipped_int':
            # For binary feature, use a bar plot instead of violin
            sns.barplot(data=plot_df, x='cluster', y=feature, hue='cluster', ax=axes[i], palette=["#3498db", "#e74c3c"], legend=False)
        else:
            # Limit duration to 95th percentile for better visualization
            if feature == 'Duration_Minutes':
                cap = plot_df[feature].quantile(0.95)
                data = plot_df[plot_df[feature] <= cap]
            else:
                data = plot_df
            sns.violinplot(data=data, x='cluster', y=feature, hue='cluster', ax=axes[i], palette=["#3498db", "#e74c3c"], legend=False)
            
        axes[i].set_title(title)
        axes[i].set_xlabel("Cluster")
        
    plt.suptitle("Feature Distributions by Cluster", fontsize=16)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Distributions Plot→ {outpath}")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(df, valid_idx, centers, label1, label2):
    plot_df = df.loc[valid_idx]
    y_true = plot_df['true_user']
    y_pred = plot_df['cluster']
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*65}")
    print(f"  TASK #3: CLUSTERING ANALYSIS ({label1} vs {label2})")
    print(f"{'='*65}")
    
    print("\n1. Are the two people's data separable?")
    print("-" * 45)
    print("Confusion Matrix (Rows: True User, Cols: Cluster):")
    print(f"             Cluster 0    Cluster 1")
    print(f"{label1:>12} {conf_matrix[0, 0]:>9}    {conf_matrix[0, 1]:>9}")
    print(f"{label2:>12} {conf_matrix[1, 0]:>9}    {conf_matrix[1, 1]:>9}")
    
    # Calculate purity
    user0_c0 = conf_matrix[0, 0]
    user0_c1 = conf_matrix[0, 1]
    user1_c0 = conf_matrix[1, 0]
    user1_c1 = conf_matrix[1, 1]
    
    total = len(plot_df)
    
    separable = (
        (user0_c0 > user0_c1 * 4 and user1_c1 > user1_c0 * 4) or
        (user0_c1 > user0_c0 * 4 and user1_c0 > user1_c1 * 4)
    )

    # Cluster-level purity percentages for the report
    c0_dominant_pct = max(user0_c0, user1_c0) / (user0_c0 + user1_c0) * 100
    c1_dominant_pct = max(user0_c1, user1_c1) / (user0_c1 + user1_c1) * 100

    print(f"\nConclusion on Separability:")
    if separable:
        print("  YES — the data is highly separable. K-Means successfully grouped")
        print("  listening sessions by user identity.")
    else:
        print("  NO — the two users are NOT easily separable.")
        print(f"  Cluster 0 is {c0_dominant_pct:.1f}% one user; Cluster 1 is {c1_dominant_pct:.1f}% one user.")
        print("  Both users' sessions are distributed across both clusters, meaning the")
        print("  algorithm grouped data by listening behaviour, not by person.")

    print("\n\n2. Explain what the clusters indicate:")
    print("-" * 45)
    print("Cluster Centers (Average feature values):")
    print(centers.round(3).to_string())

    print("\nInterpretation:")
    c0 = centers.iloc[0]
    c1 = centers.iloc[1]

    def interpret_cluster(idx, c):
        time_of_day = "Evening/Night" if c['hour'] > 16 or c['hour'] < 4 else "Morning/Daytime"
        duration = "longer" if c['Duration_Minutes'] > centers['Duration_Minutes'].mean() else "shorter"
        skips = "higher" if c['skipped_int'] > centers['skipped_int'].mean() else "lower"

        print(f"  Cluster {idx}: {time_of_day} listening sessions (avg hour {c['hour']:.1f}).")
        print(f"    - {duration} tracks (avg {c['Duration_Minutes']:.1f} min)")
        print(f"    - {skips} skip rate ({c['skipped_int']*100:.1f}%)")

    interpret_cluster(0, c0)
    print()
    interpret_cluster(1, c1)

    # Summarise the dominant difference between the two clusters
    biggest_diff_feature = (centers.iloc[0] - centers.iloc[1]).abs().idxmax()
    feature_labels = {
        'hour': 'time-of-day',
        'dayofweek': 'day-of-week',
        'Duration_Minutes': 'track duration',
        'skipped_int': 'skip behaviour',
    }
    diff_desc = feature_labels.get(biggest_diff_feature, biggest_diff_feature)

    if separable:
        print(f"\nOverall: The clusters map onto individual users — their {diff_desc}")
        print("habits are distinct enough for K-Means to tell them apart.")
    else:
        print(f"\nOverall: The clusters are primarily driven by {diff_desc}.")
        print("Both users exhibit both listening modes, so the data is not")
        print("separable by person.")
    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user1",   required=True)
    p.add_argument("--user2",   required=True)
    p.add_argument("--label1",  default="User 1")
    p.add_argument("--label2",  default="User 2")
    p.add_argument("--outdir",  default="results_clustering")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    df = load_and_prepare_data(args.user1, args.user2, args.label1, args.label2)

    features_to_cluster = ['hour', 'dayofweek', 'Duration_Minutes', 'skipped_int']
    
    print("Performing K-Means Clustering (k=2)...")
    df, centers, kmeans, scaler, valid_idx = perform_clustering(df, features_to_cluster)

    print("Generating visualizations...")
    plot_cluster_composition(df, valid_idx, args.label1, args.label2, str(out/"cluster_composition_stacked.png"))
    plot_cluster_profiles(centers, str(out/"cluster_profiles.png"))
    plot_scatter_2d(df, centers, valid_idx, str(out/"cluster_scatter_centroids.png"))
    plot_pca_scatter(df, valid_idx, kmeans, scaler, features_to_cluster, str(out/"cluster_pca_scatter.png"))
    plot_feature_distributions(df, valid_idx, str(out/"cluster_feature_distributions.png"))

    print_report(df, valid_idx, centers, args.label1, args.label2)
    print("Task 3 Clustering Complete.")

if __name__ == "__main__":
    main()
