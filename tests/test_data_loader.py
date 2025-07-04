"""
Test cases for data loading functionality.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_data, check_cached, load_cached_data, save_cached_data


class TestDataLoader:
    """Test cases for data loading functions."""
    
    def test_check_cached_missing_files(self):
        """Test check_cached with missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory should return False
            assert check_cached(tmpdir) == False
    
    def test_check_cached_all_files_present(self):
        """Test check_cached with all required files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required files
            for filename in ['X.pkl', 'y.pkl', 'subs.pkl']:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write('dummy')
            
            assert check_cached(tmpdir) == True
    
    @patch('pandas.read_excel')
    @patch('glob.glob')
    @patch('os.path.exists')
    def test_load_data_success(self, mock_exists, mock_glob, mock_read_excel):
        """Test successful data loading."""
        # Mock directory structure
        mock_glob.return_value = ['/data/sub1', '/data/sub2']
        mock_exists.return_value = True
        
        # Mock Excel file paths
        mock_glob.side_effect = [
            ['/data/sub1', '/data/sub2'],  # subject folders
            ['/data/sub1/Falls/trial1.xlsx'],  # files in Falls folder
            [],  # no files in ADLs
            [],  # no files in Near_Falls
            ['/data/sub2/ADLs/trial2.xlsx'],  # files in ADLs folder
            [],  # no files in Falls
            []   # no files in Near_Falls
        ]
        
        # Mock DataFrame
        mock_df = pd.DataFrame({'Time': [1, 2, 3], 'sensor1': [0.1, 0.2, 0.3]})
        mock_read_excel.return_value = mock_df
        
        result = load_data('/data')
        
        assert len(result) == 2
        assert result[0]['subject_id'] == 'sub1'
        assert result[0]['trial_type'] == 'Falls'
        assert result[1]['subject_id'] == 'sub2'
        assert result[1]['trial_type'] == 'ADLs'
    
    def test_load_data_no_subjects(self):
        """Test load_data with no subject folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No subject folders found"):
                load_data(tmpdir)
    
    def test_save_and_load_cached_data(self):
        """Test saving and loading cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample data
            X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            y = pd.Series([0, 1, 0])
            subs = pd.Series(['sub1', 'sub1', 'sub2'])
            
            # Save data
            save_cached_data(tmpdir, X, y, subs)
            
            # Check files were created
            assert check_cached(tmpdir) == True
            
            # Load data back
            X_loaded, y_loaded, subs_loaded = load_cached_data(tmpdir)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(X, X_loaded)
            pd.testing.assert_series_equal(y, y_loaded)
            pd.testing.assert_series_equal(subs, subs_loaded)


if __name__ == "__main__":
    pytest.main([__file__])