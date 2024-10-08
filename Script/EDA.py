import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
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
    def visualize_categorical_selected_features(self, attributes, figsize=(12, 8)):
        """
        Function to create bar plots in subplots for selected attributes.
        
        Parameters:
        attributes (list): List of columns (attributes) to plot.
        figsize (tuple): Size of the figure. Default is (10, 5).
        """
        
        # Ensure the number of attributes matches the available subplots (3 in this case)
        if len(attributes) != 3:
            print("This function is designed to handle exactly 3 attributes.")
            return
        
        # Create subplots: 2 subplots in the first row, 1 in the second
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        
        # Hide the second plot in the second row (we only need one plot in that row)
        axes[1, 1].axis('off')  # Turn off the fourth subplot
        
        # Flatten the axes for easy iteration
        axes = axes.flatten()
        
        # Loop through the selected attributes and plot
        for i, attribute in enumerate(attributes):
            self.data[attribute].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Bar plot of {attribute}')
            axes[i].set_xlabel(attribute)
            axes[i].set_ylabel('Count')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    def visualize_outliers_with_boxplot(self, attributes, figsize=(10, 5)):
        """
        Function to create box plots to detect outliers for selected attributes.
        
        Parameters:
        attributes (list): List of numerical columns (attributes) to plot box plots.
        figsize (tuple): Size of the figure. Default is (10, 5).
        """
        
        # Number of attributes
        num_attrs = len(attributes)
        
        # Dynamically determine rows and columns for subplots
        ncols = 2  # For example, we choose to have 2 columns
        nrows = (num_attrs // ncols) + (num_attrs % ncols > 0)
        
        # Create subplots for the box plots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes[1, 1].axis('off')  # Turn off the fourth subplot
        
        # Flatten axes to simplify iteration
        axes = axes.flatten() if num_attrs > 1 else [axes]
        
        # Loop through each attribute and create a box plot
        for i, attribute in enumerate(attributes):
            if i < len(axes):  # Ensure there are enough axes
                sns.boxplot(data=self.data, x=attribute, ax=axes[i])
                axes[i].set_title(f'Box plot of {attribute}')
                axes[i].set_xlabel(attribute)
            else:
                axes[i].axis('off')  # Hide unused axes if fewer attributes
        
        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()




