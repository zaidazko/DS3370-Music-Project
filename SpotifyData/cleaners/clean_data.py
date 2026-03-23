import pandas as pd
from pathlib import Path

def process_streaming_history(input_file, output_dir):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return None
    
    # drop unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])

    # convert msPlayed to Duration_Minutes
    df['Duration_Minutes'] = df['msPlayed'] / 60000

    # define skip vs play: < 30s is a skip
    df['Play Count'] = df['msPlayed'].apply(lambda x: 1 if x >= 30000 else 0)
    df['Skip Count'] = df['msPlayed'].apply(lambda x: 1 if x < 30000 else 0)

    # convert endTime to datetime
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['Date Played'] = df['endTime'].dt.date

    # rename columns to match Apple format somewhat
    df = df.rename(columns={
        'artistName': 'Artist',
        'trackName': 'Track Description'
    })

    # save
    output_path = output_dir / 'Cleaned_StreamingHistory.csv'
    df.to_csv(output_path, index=False)
    return df

def process_artist_playtime(input_file, output_dir):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return None

    # convert msPlayed to Duration_Minutes
    df['Duration_Minutes'] = df['msPlayed'] / 60000

    df = df.rename(columns={'artistName': 'Artist'})

    # save
    output_path = output_dir / 'Cleaned_Artist_Playtime.csv'
    df.to_csv(output_path, index=False)
    return df
