import pandas as pd
import gdown
import os
import shutil
import sys

def extract_data(input_file="NasData/NQ_in_daily.csv", download=True):
    # Download folder from Google Drive
    if download:
        url = "https://drive.google.com/drive/folders/13LjovZc6Vo4ZCTlyAM55vRsb4au0oJo8"
        gdown.download_folder(url, output="NasData", quiet=False, use_cookies=False)

    print(f"Extracting data from: {input_file}")
    df = pd.read_csv(input_file)

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
    if len(sys.argv) > 1:
        extract_data(sys.argv[1], download=False)
    else:
        extract_data()
