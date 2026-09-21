import pandas as pd
import gdown
import os
import shutil

def extract_data():
    # Download folder from Google Drive
    url = "https://drive.google.com/drive/folders/13LjovZc6Vo4ZCTlyAM55vRsb4au0oJo8"
    gdown.download_folder(url, output="NasData", quiet=False, use_cookies=False)

    # Use daily data
    df = pd.read_csv("NasData/NQ_in_daily.csv")

    # Rename columns to match existing pipeline
    # ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    df = df.rename(columns={
        'datetime': 'time',
        'volume': 'tick_volume'
    })

    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'], utc=True)

    # Select required columns
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    # Sort chronologically if not already
    df = df.sort_values('time').reset_index(drop=True)

    # Save to nas100_raw.csv
    df.to_csv('nas100_raw.csv', index=False)

    print("Data extraction complete. Saved to nas100_raw.csv")

if __name__ == '__main__':
    extract_data()
