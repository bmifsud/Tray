import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch

from extract_nas100_raw import extract_data

class TestExtractNas100Raw(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    @patch('gdown.download_folder')
    def test_extract_data(self, mock_gdown):
        input_csv = os.path.join(self.temp_dir.name, "NQ_in_daily.csv")
        df_in = pd.DataFrame({
            'datetime': ['2025-01-01 00:00:00', '2025-01-02 00:00:00'],
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000, 1100]
        })
        df_in.to_csv(input_csv, index=False)

        output_csv = os.path.join(self.temp_dir.name, "nas100_raw.csv")
        
        # Patch output file destination by running in temp dir or mocking to_csv
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            extract_data(input_file=input_csv, download=True)
            mock_gdown.assert_called_once()
            mock_to_csv.assert_called_with('nas100_raw.csv', index=False)

if __name__ == '__main__':
    unittest.main()
