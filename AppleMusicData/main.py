from pathlib import Path

# we only need to run main.py to clean data or if we have new data 


# importing cleaners
from cleaners.clean_daily import process_daily_tracks
from cleaners.clean_containers import process_containers

# path setups
base_dir = Path(__file__).parent
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed'

# getting file paths 
daily_tracks_file = raw_dir / 'Apple Music - Play History Daily Tracks.csv' # useful for finding out what songs the user has played (contains ~200k rows)
containers_file = raw_dir / 'Apple Music - Container Details.csv' # useful for finding out what playlists the user has created (contains ~2000 rows)

if __name__ == '__main__':
    
    # run the cleaners
    df_tracks = process_daily_tracks(daily_tracks_file, processed_dir)
    df_containers = process_containers(containers_file, processed_dir)
    print("data cleaned")
    