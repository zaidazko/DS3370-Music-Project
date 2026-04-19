import pandas as pd
from pathlib import Path

def generate_summary_statistics():
    base_dir = Path(__file__).parent
    processed_dir = base_dir / 'data' / 'processed'
    
    try:
        df_tracks = pd.read_csv(processed_dir / 'Cleaned_StreamingHistory.csv')
    except FileNotFoundError:
        print("Error: Could not find clean data. Make sure you have run main.py first!")
        return

    # calculate core metrics
    total_hours = df_tracks['Duration_Minutes'].sum() / 60
    unique_tracks = df_tracks['Track Description'].nunique()
    unique_artists = df_tracks['Artist'].nunique()
    
    # calculate play vs. skip metrics
    total_plays = df_tracks['Play Count'].sum()
    total_skips = df_tracks['Skip Count'].sum()
    total_interactions = total_plays + total_skips
    skip_rate = (total_skips / total_interactions) * 100 if total_interactions > 0 else 0
    
    # calculate daily averages
    daily_listening = df_tracks.groupby('Date Played')['Duration_Minutes'].sum() / 60
    avg_daily_hours = daily_listening.mean()
    busiest_day_date = daily_listening.idxmax()
    busiest_day_hours = daily_listening.max()
    
    # calculate top artist
    top_artist = df_tracks.groupby('Artist')['Duration_Minutes'].sum().idxmax()
    top_artist_hours = df_tracks.groupby('Artist')['Duration_Minutes'].sum().max() / 60

    # print the final report
    print(f"Total Listening Time:     {total_hours:,.1f} hours")
    print(f"Average Daily Listening:  {avg_daily_hours:.1f} hours/day")
    print(f"Busiest Single Day:       {busiest_day_date} ({busiest_day_hours:.1f} hours)")
    print("-" * 40)
    print(f"Unique Tracks Played:     {unique_tracks:,}")
    print(f"Unique Artists Played:    {unique_artists:,}")
    print(f"Overall Top Artist:       {top_artist} ({top_artist_hours:.1f} hours)")
    print("-" * 40)
    print(f"Total Completed Plays:    {total_plays:,}")
    print(f"Total Skipped Tracks:     {total_skips:,}")
    print(f"Overall Skip Rate:        {skip_rate:.1f}%")
    print("-" * 40)

    # print some rankings
    print("Rankings:")
    print("-" * 40)
    print("Top 10 Artists:")
    artist_totals = df_tracks.groupby('Artist')['Duration_Minutes'].sum().sort_values(ascending=False).head(10)
    artist_total_hours = artist_totals / 60
    for i, (artist, hours) in enumerate(artist_total_hours.items(), start=1):
        print(f"{i}| {artist}, {hours:.2f} hours")

    print("\nTop 10 Tracks:")
    print("-" * 40)
    track_totals = df_tracks.groupby('Track Description')['Play Count'].sum().sort_values(ascending=False).head(10)
    for i, (track, count) in enumerate(track_totals.items(), start=1):
        print(f"{i}| {track}: {count} plays")

if __name__ == '__main__':
    generate_summary_statistics()