import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class Eda:
    def __init__(self, data):
        self.data = data
    ### summery description of data for numeric attributes 
    def summary(self):
        # Summary for numeric columns
        numeric_summary = self.data.select_dtypes(include=['float64', 'int64']).describe().T
        print("Numeric Columns Summary:")
        print(numeric_summary)
        print("\n")  # Print a newline for better formatting

        # Summary for object (categorical) columns
        object_summary = self.data.select_dtypes(include=['object']).describe().T
        print("Object Columns Summary:")
        print(object_summary)
    # Check data types of each column
    def check_datatypes(self):
        print(self.data.dtypes)
    # check number of row and col
    def no_col_row(self):
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")
    # check the info
    def info(self):
        print(self.data.info())
    # check missing values
    def missing_values(self):
        print(self.data.isnull().sum())
    def correlation_num_col(self):
        selec_futu = self.data.drop(columns=['CountryCode'], errors='ignore')
        numeric_cols = selec_futu.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = selec_futu[numeric_cols].corr()
        return corr_matrix  # Return the correlation matrix
    def corr_heatmap(self):
        corr_matrix = self.correlation_num_col()
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    def visualize_numerical_features(self):
        # Select numerical columns
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns

        # Set up the matplotlib figure
        num_features = len(numeric_cols)
        plt.figure(figsize=(15, num_features * 5))

        for i, col in enumerate(numeric_cols):
            plt.subplot(num_features, 2, i * 2 + 1)  # Histogram
            sns.histplot(self.data[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

            plt.subplot(num_features, 2, i * 2 + 2)  # Box plot
            sns.boxplot(x=self.data[col])
            plt.title(f'Box Plot of {col}')

        plt.tight_layout()
        plt.show()