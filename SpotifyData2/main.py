from pathlib import Path
from cleaners.clean_data import process_streaming_history

# path setups
base_dir = Path(__file__).parent
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed'

streaming_file = raw_dir / 'spotify_history.csv'

if __name__ == '__main__':
    df_streaming = process_streaming_history(streaming_file, processed_dir)
    print("data cleaned")
