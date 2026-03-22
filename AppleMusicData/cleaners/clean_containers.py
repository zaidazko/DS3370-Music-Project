import pandas as pd

def process_containers(raw_path, processed_dir):
    df = pd.read_csv(raw_path)
    
    # Filter columns
    cols_to_keep = ['Container Description', 'Container Type', 'Genres', 'Artists', 'Play Count', 'Play Duration Milliseconds']
    clean_df = df[cols_to_keep].copy()
    
    # Clean data (Convert duration)
    clean_df['Duration_Hours'] = clean_df['Play Duration Milliseconds'] / (1000 * 60 * 60)
    clean_df = clean_df.drop(columns=['Play Duration Milliseconds'])
    
    # Save it
    output_path = processed_dir / 'Cleaned_Containers.csv'
    clean_df.to_csv(output_path, index=False)
    return clean_df