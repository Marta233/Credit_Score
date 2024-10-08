import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class RFMS:
    def __init__(self, df):
        self.df = df
        self.rfms = None
        self.rfms_score = None
    def outlier_percentage(self):
        # Filter for numeric columns
        numeric_df = self.df.select_dtypes(include=[float, int])

        # Ensure there is at least one numeric column
        if numeric_df.empty:
            raise ValueError("No numeric columns available for outlier detection.")

        # Identify outliers using the IQR method
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate outliers for each column
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))

        # Count total values per column and calculate the percentage of outliers
        total_values = numeric_df.count()  # Non-NaN values in each column
        outlier_percentage = (outliers.sum() / total_values) * 100

        # Create a DataFrame for outlier percentages
        outlier_df = pd.DataFrame({
            'Column': numeric_df.columns,
            'outlier_percentage': outlier_percentage
        }).sort_values(by='outlier_percentage', ascending=False)

        return outlier_df

    def cap_outliers(self):
            """
            Cap the outliers to the lower and upper bounds using the IQR method.
            """
            # Filter for numeric columns
            numeric_df = self.df.select_dtypes(include=[float, int])

            # Ensure there is at least one numeric column
            if numeric_df.empty:
                raise ValueError("No numeric columns available for outlier capping.")

            # Identify outliers using the IQR method
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1

            # Define the lower and upper bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap the outliers to the lower and upper bounds
            for column in numeric_df.columns:
                # Cap lower outliers
                self.df[column] = self.df[column].clip(lower=lower_bound[column], upper=upper_bound[column])

            return self.df
    
    def Customer_RFMS(self):
        # Convert TransactionStartTime to datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

        # Recency: Days since the last transaction for each CustomerId
        self.df['Recency'] = (self.df['TransactionStartTime'].max() - self.df['TransactionStartTime']).dt.days
        
        # For customers with multiple transactions, take the minimum Recency (most recent transaction)
        recency = self.df.groupby('CustomerId')['Recency'].min().reset_index()
        recency.columns = ['CustomerId', 'Recency']
        
        # Frequency: Count the number of transactions for each CustomerId
        frequency = self.df.groupby('CustomerId')['TransactionId'].count().reset_index()
        frequency.columns = ['CustomerId', 'Frequency']
        
        # Monetary: Sum of the transaction amounts for each CustomerId
        monetary = self.df.groupby('CustomerId')['Amount'].sum().reset_index()
        monetary.columns = ['CustomerId', 'MonetaryValue']
        
        # Merge Recency, Frequency, and Monetary
        rfms = pd.merge(frequency, monetary, on='CustomerId')
        rfms = pd.merge(rfms, recency, on='CustomerId')

        # Create RFMS Score
        rfms['RFMS_Score'] = rfms[['Recency', 'Frequency', 'MonetaryValue']].sum(axis=1)

        self.rfms = rfms  # Store RFMS result in the class attribute
        # Call merge_two_data to merge RFMS with demographics
        data1 =pd.read_csv('../Data/data_v1.csv')
        merged_rfms = self.merge_two_data(rfms, data1)

        self.rfms = merged_rfms  # Store the merged RFMS result in the class attribute
        return merged_rfms
    def customer_score_RFMS(self):
        scaler = StandardScaler()
        self.rfms['RFMS_Score_Standardized'] = scaler.fit_transform(self.rfms[['RFMS_Score']])
        
        # Calculate the median score to use as a threshold
        threshold = 0.01
        # Classify users as Good (high RFMS) or Bad (low RFMS)
        self.rfms['CreditRiskLabel'] = np.where(self.rfms['RFMS_Score_Standardized'] >= threshold, 'Good', 'Bad')
        
        return self.rfms

    def Plot_RFMS_distribution(self):
        # Plotting the distribution of RFMS Score
        if self.rfms is None:
            print("RFMS score has not been calculated. Please run Customer_RFMS first.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(self.rfms['RFMS_Score'], bins=20, alpha=0.7, color='blue', label='RFMS Score')
        plt.title('RFMS Score Distribution')
        plt.xlabel('RFMS Score')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
    def plot_creditrisklabel(self):
        if self.rfms is None:
            print("RFMS score has not been calculated. Please run Customer_RFMS first.")
            return

        plt.figure(figsize=(10, 6))
        self.rfms['CreditRiskLabel'].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.title('Credit Risk Label Distribution')
        plt.xlabel('Credit Risk Label')
        plt.ylabel('Count')
        plt.show()
    def Plot_RFMS_distribution(self):
        if self.rfms is None:
            print("RFMS score has not been calculated. Please run Customer_RFMS first.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(self.rfms['RFMS_Score_Standardized'], bins=30, alpha=0.7, color='blue', label='Standardized RFMS Score')
        plt.title('Standardized RFMS Score Distribution')
        plt.xlabel('Standardized RFMS Score')
        plt.ylabel('Count')
        plt.axvline(x=0, color='red', linestyle='--', label='Initial Threshold')
        plt.legend()
        plt.show()
    def merge_two_data(self, data1,data2):
        # Merge two datasets based on CustomerId
        merged_data = pd.merge(data1, data2, on='CustomerId', how='inner')
        if 'Unnamed: 0' in merged_data.columns:
            merged_data.drop(columns=['Unnamed: 0'], inplace=True)
        return merged_data

