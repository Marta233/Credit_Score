import pandas as pd
import numpy as np
from monotonic_binning.monotonic_woe_binning import Binning
import scorecardpy as sc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class WOEAnalysis:
    def __init__(self, df, target):
        self.df = df.copy()  # Avoid modifying the original DataFrame
        self.target = target
        self.breaks = {}
        self.scaler = StandardScaler()  # Initialize the scaler
        self._clean_data()

    def _clean_data(self):
        """Clean the DataFrame by dropping unnecessary columns."""
        if 'CustomerId' in self.df.columns:
            self.df = self.df.drop('CustomerId', axis=1)

    def encode_target(self):
        """Encode target variable as numeric values and retain a mapping."""
        if self.target in self.df.columns:
            # Get unique values and create a mapping
            unique_values = self.df[self.target].unique()
            
            # Create a mapping dictionary
            self.target_mapping = {value: idx for idx, value in enumerate(unique_values)}
            
            # Encode the target variable
            self.df[self.target] = self.df[self.target].map(self.target_mapping)
        else:
            raise ValueError(f"Target variable '{self.target}' not found in DataFrame.")

    def scale_numerical_features(self):
        """Scale numerical features using StandardScaler."""
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.drop(self.target)
        
        # Fit and transform the numerical features
        self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])
        print("Scaled Numerical Features:")
        print(self.df[numerical_features].head())

    def woe_num(self):
        """Calculate breaks for numerical features using monotonic binning."""
        self.encode_target()  # Encode target variable
        self.scale_numerical_features()  # Scale numerical features
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.drop(self.target)

        for col in numerical_features:
            bin_object = Binning(self.target, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
            bin_object.fit(self.df[[self.target, col]])
            self.breaks[col] = bin_object.bins[1:-1].tolist()  # Exclude the first and last break

        return self.breaks

    def adjust_woe(self):
        """Adjust the WoE calculation and plot the results."""
        if not self.breaks:
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")

        bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')  # Use '1' for "Bad"
        
        # Display the adjusted bins
        print("Adjusted Binning Results:")
        print(bins_adj)
        # Assume bins_adj is already created
        plt.figure(figsize=(12, 8))  # Set the figure size
        sc.woebin_plot(bins_adj)  # Create the WoE plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.title("WoE Binning Plot")  # Optional: Add a title
        plt.tight_layout()  # Adjust layout
        plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin if needed
        plt.show()  # Display the plot

    def woeval(self, train):
        """Convert a DataFrame into WoE values based on adjusted bins."""
        self.encode_target()  # Encode target variable
        if not self.breaks:
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")
        
        if not isinstance(train, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')
        train_woe = sc.woebin_ply(train, bins_adj)
        return train_woe

    def calculate_iv(self, df_merged, y):
        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
        
        # Step 1: Map 'CreditRiskLabel' to numeric values
        risk_mapping = {'Good': 1, 'Bad': 0}  # Define your mapping
        df_merged['CreditRiskLabel'] = df_merged['CreditRiskLabel'].map(risk_mapping)
        df_merged = df_merged
        # Step 2: Remove 'CustomerId' and 'CreditRiskLabel' from the DataFrame
        df_merged1 = df_merged.drop(['CustomerId'], axis=1)
        # Calculate Information Value (IV)
        iv_results = sc.iv(df_merged1, y=y)  # Ensure 'sc' is correctly imported
        return df_merged, iv_results