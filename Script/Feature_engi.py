import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Feature_Engineering:
    def __init__(self, data):
        self.data = data
        self.label_encoders = {}  # To store label encoders for each attribute

    def onehotencoding_selected_col(self, columns):
        """
        Perform one-hot encoding on selected columns and merge back into the original DataFrame.
        
        :param columns: List of columns to apply one-hot encoding to
        """
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")

        # Perform one-hot encoding without dropping the first category
        encoded_df = pd.get_dummies(self.data[columns], drop_first=False, prefix=columns)

        # Drop original columns and concatenate the new encoded columns
        self.data = pd.concat([self.data.drop(columns, axis=1), encoded_df], axis=1)
        
        print("One-hot encoding completed for columns:", columns)
        return self.data
    def outlier_percentage_all_col(self, remove_outliers=True):
        # Filter for numeric columns
        numeric_df = self.data.select_dtypes(include=[float, int])

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

        # If remove_outliers is set to True, drop the rows with outliers
        if remove_outliers:
            # Use the outliers DataFrame to filter the rows
            no_outlier_df = self.data[~outliers.any(axis=1)]  # Remove rows where any column has an outlier
            
            print(f"Removed {len(self.data) - len(no_outlier_df)} rows with outliers.")
            self.data = no_outlier_df  # Update self.df with the outlier-removed data
            
            return no_outlier_df

        # Otherwise, just return the outlier percentages
        return outlier_df
    def outlier_percentage(self):
        # Filter for numeric columns
        numeric_df = self.data.select_dtypes(include=[float, int])

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
            numeric_df = self.data.select_dtypes(include=[float, int])

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
                self.data[column] = self.data[column].clip(lower=lower_bound[column], upper=upper_bound[column])

            return self.data

    def feature_engineering_transaction(self):
        # Create Debit and Credit columns
        self.data['Debit'] = self.data['Amount'].where(self.data['Amount'] < 0, 0).abs()   # Positive values for debits
        self.data['Credit'] = self.data['Amount'].where(self.data['Amount'] > 0, 0)        # Positive values for credits

        # Group by CustomerId and calculate aggregate features
        self.data = self.data.groupby('CustomerId').agg(
            Total_Credit_Amount=('Credit', 'sum'),
            Total_Debit_Amount=('Debit', 'sum'),
            Total_Transaction_amount=('Value', 'sum'),
            Average_Transaction_Amount=('Value', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Total_Unique_Account_Count=('AccountId', 'nunique'),
            FraudResult=('FraudResult', 'max'),  # Returns 1 if any of the values is 1 (indicating fraud)
            ChannelId_ChannelId_1=('ChannelId_ChannelId_1', 'sum'),  # Sum up True values per ChannelId column
            ChannelId_ChannelId_2=('ChannelId_ChannelId_2', 'sum'),
            ChannelId_ChannelId_3=('ChannelId_ChannelId_3', 'sum'),
            ChannelId_ChannelId_5=('ChannelId_ChannelId_5', 'sum'),
            ProductCategory_airtime = ('ProductCategory_airtime', 'sum'),
            ProductCategory_data_bundles = ('ProductCategory_data_bundles', 'sum'),
            ProductCategory_financial_services = ('ProductCategory_financial_services','sum'),
            ProductCategory_movies=('ProductCategory_movies', 'sum'),
            ProductCategory_other=('ProductCategory_other','sum') ,
            ProductCategory_ticket=('ProductCategory_ticket','sum'),
            ProductCategory_transport = ('ProductCategory_transport','sum'),
            ProductCategory_tv =('ProductCategory_tv','sum'),
            ProductCategory_utility_bill = ('ProductCategory_utility_bill', 'sum'),
            Standard_Deviation_Transaction_Amount=('Amount', lambda x: x.std(ddof=0) if x.count() > 1 else 0)
        ).reset_index()
        return self.data
    def extract_transaction_features(self):
        # Extract features like day, month, and year from TransactionStartTime
        if 'TransactionStartTime' not in self.data.columns:
            raise KeyError("The column 'TransactionStartTime' does not exist in the DataFrame.")
        
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        
        # Extract features
        self.data['Transaction_Hour'] = self.data['TransactionStartTime'].dt.hour
        self.data['Transaction_Day'] = self.data['TransactionStartTime'].dt.day
        self.data['Transaction_Month'] = self.data['TransactionStartTime'].dt.month
        self.data['Transaction_Year'] = self.data['TransactionStartTime'].dt.year
        
        return self.data

    def remove_some_cols(self, attrib):
        # Remove specific columns
        self.data = self.data.drop(attrib, axis=1)
        return self.data

    def numeric_feature_scaling(self):
        # Create a StandardScaler object
        scaler = StandardScaler()
        
        # Select only the specified numeric attributes, excluding 'CustomerId'
        selected_data = self.data.drop(columns=['CustomerId'], errors='ignore')

        # Fit the scaler to the data and transform it
        scaled_data = scaler.fit_transform(selected_data)

        # Convert the scaled data back into a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=selected_data.columns)

        # Assign the scaled data back to the original dataframe for the selected attributes
        self.data[scaled_df.columns] = scaled_df

        # Print the scaled dataframe for confirmation
        print(scaled_df)
        return self.data
