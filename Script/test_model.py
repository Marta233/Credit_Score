import pandas as pd
import joblib

def calculate_credit_score(probability):
    """
    Calculate the credit score based on the given probability.

    Parameters:
    - probability: float, the probability of the positive class (class 1).

    Returns:
    - score: float, the calculated credit score.
    """
    Score_min = 300
    Score_max = 900
    score = Score_min + (1 - probability) * (Score_max - Score_min)
    return score

def predict_probabilities(test_data, model_path, target_column=None):
    """
    Load the saved model and predict probabilities on the provided test dataset.

    Parameters:
    - test_data: pd.DataFrame, the test dataset as a DataFrame.
    - model_path: str, path to the saved model file.
    - target_column: str or None, name of the target column to drop from test data if it exists.

    Returns:
    - pd.DataFrame, predicted probabilities of class 1 with CustomerId and credit scores.
    """
    # Load the saved model
    model = joblib.load(model_path)

    # Get the feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = set(model.feature_names_in_)  # Use set for intersection
    else:
        feature_names = set(model.get_feature_names_out())

    # Prepare the test data by dropping unnecessary columns
    if target_column and target_column in test_data.columns:
        test_data = test_data.drop(columns=[target_column])

    # Drop 'CustomerId' from the test data for predictions
    test_data_filtered = test_data.drop(columns=['CustomerId'], errors='ignore')

    # Select only the features that were used to fit the model
    test_data_filtered = test_data_filtered[test_data_filtered.columns.intersection(feature_names)]

    # Store CustomerId separately
    customer_id = test_data['CustomerId'] if 'CustomerId' in test_data.columns else None

    # Predict probabilities on the filtered test data
    predicted_probabilities = model.predict_proba(test_data_filtered)

    # Extract probabilities for class 1 (positive class)
    results_df = pd.DataFrame({
        'CustomerId': customer_id.values if customer_id is not None else None,
        'Prob_Class_1': predicted_probabilities[:, 1]  # Only class 1 probabilities
    })

    # Calculate credit scores based on the probabilities
    results_df['Credit_Score'] = results_df['Prob_Class_1'].apply(calculate_credit_score)

    # Display the predicted probabilities for class 1 and credit scores
    print("Predicted Probabilities for Class 1 and Credit Scores:")
    print(results_df)

    return results_df
