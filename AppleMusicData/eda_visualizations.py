import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pandas.plotting import parallel_coordinates
from pathlib import Path

def generate_visualizations():
    print("--- GENERATING VISUALIZATIONS (PART 1C) ---\n")
    
    # setup paths
    base_dir = Path(__file__).parent
    processed_dir = base_dir / 'data' / 'processed'
    plots_dir = base_dir / 'plots'
    
    # load the cleaned up data
    try:
        df_tracks = pd.read_csv(processed_dir / 'Cleaned_Daily_Tracks.csv')
        df_containers = pd.read_csv(processed_dir / 'Cleaned_Containers.csv')
    except FileNotFoundError:
        print("Error: Could not find clean data. Run main.py first!")
        return

    # extract artist and convert dates
    df_tracks['Date Played'] = pd.to_datetime(df_tracks['Date Played'])
    df_tracks['Artist'] = df_tracks['Track Description'].apply(
        lambda x: str(x).split(' - ')[0] if ' - ' in str(x) else 'Unknown'
    )
    
    # set a clean visual style for all charts
    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # histogram (Distribution of Daily Listening)
    # ---------------------------------------------------------
    print("Generating Histogram...")
    daily_listening = df_tracks.groupby('Date Played')['Duration_Minutes'].sum() / 60
    
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_listening, bins=30, color='skyblue', edgecolor='black', kde=True)
    plt.title('Distribution of Daily Listening Time')
    plt.xlabel('Hours Listened per Day')
    plt.ylabel('Frequency (Number of Days)')
    plt.tight_layout()
    plt.savefig(plots_dir / '1_histogram_daily_listening.png')
    plt.close()

    # ---------------------------------------------------------
    # line chart (Listening Trends Over Time)
    # ---------------------------------------------------------
    print("Generating Line Chart...")
    plt.figure(figsize=(12, 6))
    
    # group by month to make the line chart cleaner (M stands for Month End)
    monthly_listening = df_tracks.set_index('Date Played').resample('ME')['Duration_Minutes'].sum() / 60
    
    sns.lineplot(x=monthly_listening.index, y=monthly_listening.values, marker='o', color='coral', linewidth=2)
    plt.title('Total Listening Hours per Month')
    plt.xlabel('Date')
    plt.ylabel('Total Hours')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / '2_line_chart_monthly_trend.png')
    plt.close()

    # ---------------------------------------------------------
    # box plot (Listening Sessions for Top 5 Artists)
    # ---------------------------------------------------------
    print("Generating Box Plot...")
    plt.figure(figsize=(12, 6))
    
    # get top 5 artists by total time
    top_5_artists = df_tracks.groupby('Artist')['Duration_Minutes'].sum().nlargest(5).index
    top_5_df = df_tracks[df_tracks['Artist'].isin(top_5_artists)]
    
    sns.boxplot(x='Duration_Minutes', y='Artist', data=top_5_df, order=top_5_artists, palette='Set2')
    plt.title('Distribution of Individual Listening Session Lengths (Top 5 Artists)')
    plt.xlabel('Session Duration (Minutes)')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.savefig(plots_dir / '3_box_plot_top_artists.png')
    plt.close()

    # ---------------------------------------------------------
    # word cloud (Top Genres)
    # ---------------------------------------------------------
    print("Generating Word Cloud...")
    # clean up the comma-separated genres from the containers file
    genres_series = df_containers['Genres'].dropna().str.split(',').explode().str.strip()
    genre_counts = genres_series.value_counts().to_dict()
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate_from_frequencies(genre_counts)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Listened To Genres')
    plt.tight_layout()
    plt.savefig(plots_dir / '4_wordcloud_genres.png')
    plt.close()

    # ---------------------------------------------------------
    # parallel coordinates plot (Top 25 Tracks Metrics)
    # ---------------------------------------------------------
    print("Generating Parallel Coordinates Plot...")
    # aggregate stats per track
    track_stats = df_tracks.groupby('Track Description').agg({
        'Play Count': 'sum',
        'Skip Count': 'sum',
        'Duration_Minutes': 'sum'
    }).reset_index()
    
    top_25_tracks = track_stats.nlargest(25, 'Duration_Minutes').copy()
    
    # normalize metrics to be between 0 and 1 so they fit on the same graph
    for col in ['Play Count', 'Skip Count', 'Duration_Minutes']:
        top_25_tracks[f'{col}_Norm'] = (top_25_tracks[col] - top_25_tracks[col].min()) / (top_25_tracks[col].max() - top_25_tracks[col].min())
        
    parallel_df = top_25_tracks[['Track Description', 'Play Count_Norm', 'Skip Count_Norm', 'Duration_Minutes_Norm']]
    
    plt.figure(figsize=(12, 6))
    parallel_coordinates(parallel_df, 'Track Description', colormap='tab20', alpha=0.7)
    plt.title('Parallel Coordinates: Top 25 Tracks (Play vs Skip vs Duration)')
    plt.ylabel('Normalized Score (0 to 1)')
    plt.legend().set_visible(False) # hide legend because 25 tracks makes it too messy
    plt.tight_layout()
    plt.savefig(plots_dir / '5_parallel_coordinates.png')
    plt.close()

    print("\nSuccess! All 5 charts have been saved to the 'plots/' folder.")

if __name__ == '__main__':
    generate_visualizations()