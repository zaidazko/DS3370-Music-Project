import pandas as pd

def process_daily_tracks(raw_path, processed_dir):
    df = pd.read_csv(raw_path)
    
    # Filter columns
    cols_to_keep = ['Date Played', 'Track Description', 'Play Duration Milliseconds', 'Play Count', 'Skip Count', 'End Reason Type']
    clean_df = df[cols_to_keep].copy()
    
    # Clean data
    clean_df['Duration_Minutes'] = clean_df['Play Duration Milliseconds'] / 60000
    clean_df['Date Played'] = pd.to_datetime(clean_df['Date Played'], format='%Y%m%d')
    clean_df = clean_df.drop(columns=['Play Duration Milliseconds'])
    
    # Save it
    output_path = processed_dir / 'Cleaned_Daily_Tracks.csv'
    clean_df.to_csv(output_path, index=False)
    return clean_df