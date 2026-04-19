import pandas as pd
from pathlib import Path

def process_streaming_history(input_file, output_dir):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return None
    
    # Check for empty index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif len(df.columns) > 0 and df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])

    # Map columns to match the previous SpotifyData schema
    column_mapping = {
        'ts': 'endTime',
        'artist_name': 'Artist',
        'track_name': 'Track Description',
        'ms_played': 'msPlayed'
    }
    df = df.rename(columns=column_mapping)

    # In case msPlayed or others are not present due to weird files, ensure we have them
    if 'msPlayed' in df.columns:
        # convert msPlayed to Duration_Minutes
        df['Duration_Minutes'] = df['msPlayed'] / 60000

        # define skip vs play
        # Use 'skipped' column if available, else fallback to < 30s logic
        if 'skipped' in df.columns:
            # skipped might be boolean or string ('TRUE'/'FALSE')
            # Let's map it cleanly:
            df['skipped'] = df['skipped'].astype(str).str.upper() == 'TRUE'
            df['Skip Count'] = df['skipped'].astype(int)
            df['Play Count'] = (~df['skipped']).astype(int)
        else:
            df['Play Count'] = df['msPlayed'].apply(lambda x: 1 if x >= 30000 else 0)
            df['Skip Count'] = df['msPlayed'].apply(lambda x: 1 if x < 30000 else 0)

    if 'endTime' in df.columns:
        # convert endTime to datetime
        df['endTime'] = pd.to_datetime(df['endTime'])
        df['Date Played'] = df['endTime'].dt.date

    # Keep all other columns (like platform, album_name, reason_start) just in case,
    # as they provide rich features for clustering/outliers, but the original ones are present.

    # save
    output_path = output_dir / 'Cleaned_StreamingHistory.csv'
    df.to_csv(output_path, index=False)
    return df
