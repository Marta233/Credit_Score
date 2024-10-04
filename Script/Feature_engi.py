import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class Feature_Engineering:
    def __init__(self, data):
        self.data = data
        self.label_encoders = {}  # To store label encoders for each attribute

    def feature_engineering_transaction(self):
        # Create Debit and Credit columns
        self.data['Debit'] = self.data['Amount'].where(self.data['Amount'] < 0, 0).abs()   # Positive values for debits
        self.data['Credit'] = self.data['Amount'].where(self.data['Amount'] > 0, 0)        # Positive values for credits
        
        # Group by CustomerId to calculate total credit and debit amounts, average transaction amount, transaction count, and unique account coun
        self.data = self.data.groupby('CustomerId').agg(
            Total_Credit_Amount=('Credit', 'sum'),
            Total_Debit_Amount=('Debit', 'sum'),
            Average_Transaction_Amount=('Value', 'mean'),  # Assuming 'Amount' is the column that exists in your DataFrame
            Transaction_Count=('TransactionId', 'count'),
            Total_Unique_Account_Count=('AccountId', 'nunique'),
            distinct_product_categories=('ProductCategory', 'nunique'),
            Standard_Deviation_Transaction_Amount=('Amount', lambda x: x.std(ddof=0) if x.count() > 1 else 0)  # Replace NaN with 0
        ).reset_index()
        return self.data
    def extract_transaction_features(self):
        if 'TransactionStartTime' not in self.data.columns:
            raise KeyError("The column 'TransactionStartTime' does not exist in the DataFrame.")
        
        # Convert 'TransactionStartTime' to datetime
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        
        # Extract features
        self.data['Transaction_Hour'] = self.data['TransactionStartTime'].dt.hour
        self.data['Transaction_Day'] = self.data['TransactionStartTime'].dt.day
        self.data['Transaction_Month'] = self.data['TransactionStartTime'].dt.month
        self.data['Transaction_Year'] = self.data['TransactionStartTime'].dt.year
        
        return self.data
    def remove_some_cols(self, attrib):
        self.data = self.data.drop(attrib, axis=1)
        return self.data
    def label_encode(self, attribute):
        """
        Applies label encoding to a specified attribute.
        
        Parameters:
        attribute (str): The name of the attribute to encode.
        """
        if attribute in self.data.columns:
            le = LabelEncoder()
            self.data[attribute] = le.fit_transform(self.data[attribute].astype(str))  # Convert to string to avoid numeric misinterpretation
            self.label_encoders[attribute] = le  # Store the encoder for potential inverse transformation
        else:
            raise ValueError(f"Attribute '{attribute}' not found in the DataFrame.")

    def label_encoding_selected_attribute(self, attributes):
        """
        Applies label encoding to a list of specified attributes.
        
        Parameters:
        attributes (list): The list of attribute names to encode.
        """
        for attribute in attributes:
            self.label_encode(attribute)
        return self.data
    def numeric_feature_scaling(self, attributes):
        # Create a StandardScaler object
        scaler = StandardScaler()
        
        # Select only the specified numeric attributes
        selected_data = self.data[attributes]

        # Fit the scaler to the data and transform it
        scaled_data = scaler.fit_transform(selected_data)

        # Convert the scaled data back into a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=selected_data.columns)

        # Assign the scaled data back to the original dataframe for the selected attributes
        self.data[attributes] = scaled_df

        # Print the scaled dataframe for confirmation
        print(scaled_df)