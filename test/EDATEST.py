import unittest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/10 A KAI 2/Week6/Credit_Scoring/')
from Script.EDA import Eda
class TestEda(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample dataframe for testing
        cls.data = pd.DataFrame({
            'numeric_col1': [1, 2, 3, 4, 5],
            'numeric_col2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'object_col': ['A', 'B', 'A', 'C', 'B'],
            'CountryCode': ['ET', 'ET', 'ET', 'ET', 'ET']
        })
        cls.eda = Eda(cls.data)

    def test_summary(self):
        # Test summary for numeric and object columns
        self.eda.summary()

    def test_check_datatypes(self):
        # Test to check data types of columns
        self.eda.check_datatypes()

    def test_no_col_row(self):
        # Test to check the number of rows and columns
        self.eda.no_col_row()

    def test_info(self):
        # Test to check the info of the dataset
        self.eda.info()

    def test_missing_values(self):
        # Test for missing values
        self.eda.missing_values()

    def test_correlation_num_col(self):
        # Test correlation matrix
        corr_matrix = self.eda.correlation_num_col()
        self.assertIsInstance(corr_matrix, pd.DataFrame)

    def test_corr_heatmap(self):
        # Test correlation heatmap (no return value, just checks execution)
        self.eda.corr_heatmap()

    def test_visualize_numerical_features(self):
        # Test visualization of numerical features
        self.eda.visualize_numerical_features()

    def test_visualize_categorical_selected_features(self):
        # Test visualization of selected categorical features
        attributes = ['object_col', 'CountryCode', 'numeric_col1']
        self.eda.visualize_categorical_selected_features(attributes)

    def test_visualize_outliers_with_boxplot(self):
        # Test visualization of outliers with boxplots
        attributes = ['numeric_col1', 'numeric_col2']
        self.eda.visualize_outliers_with_boxplot(attributes)

if __name__ == '__main__':
    unittest.main()
