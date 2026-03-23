from pathlib import Path
from cleaners.clean_data import process_streaming_history, process_artist_playtime

# path setups
base_dir = Path(__file__).parent
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed'

streaming_file = raw_dir / 'StreamingHistory_music.csv'
playtime_file = raw_dir / 'spotify_artist_playtime.csv'

if __name__ == '__main__':
    df_streaming = process_streaming_history(streaming_file, processed_dir)
    df_playtime = process_artist_playtime(playtime_file, processed_dir)
    print("data cleaned")
