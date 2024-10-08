import pandas as pd
import joblib

def calculate_credit_score(probability: float) -> float:
    Score_min = 300
    Score_max = 900
    score = Score_min + (1 - probability) * (Score_max - Score_min)
    return score

def get_credit_score_label(score: float) -> int:
    if score < 601:
        return 1  # High Risk
    elif 601 <= score < 701:
        return 2  # Moderate Risk
    elif 701 <= score < 801:
        return 3  # Low Risk
    elif 801 <= score <= 900:
        return 4  # Very Low Risk
    else:
        raise ValueError("Score out of range")

def predict_probabilities(test_data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    # Load the saved model
    model = joblib.load(model_path)

    # Get the feature names used during model training
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        feature_names = model.get_feature_names_out()

    # Drop 'CustomerId' from the test data for predictions
    test_data_filtered = test_data.drop(columns=['CustomerId'], errors='ignore')

    # Ensure the test dataset only contains features present in the model's training
    missing_features = set(feature_names) - set(test_data_filtered.columns)
    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")

    # Filter the test dataset to include only the necessary features in the correct order
    test_data_filtered = test_data_filtered[feature_names]

    # Store CustomerId separately
    customer_id = test_data['CustomerId'] if 'CustomerId' in test_data.columns else None

    # Predict probabilities on the filtered test data
    predicted_probabilities = model.predict_proba(test_data_filtered)

    # Create DataFrame with predicted probabilities
    results_df = pd.DataFrame({
        'CustomerId': customer_id.values if customer_id is not None else None,
        'Prob Good Credit': predicted_probabilities[:, 1]  # Probability of class 1
    })

    # Calculate credit scores based on the probabilities
    results_df['Credit_Score'] = results_df['Prob Good Credit'].apply(calculate_credit_score)

    # Assign credit score labels
    results_df['Credit_Score_Label'] = results_df['Credit_Score'].apply(get_credit_score_label)

    # Display the predicted probabilities for class 1, credit scores, and labels
    print("Predicted Probabilities for Class 1, Credit Scores, and Labels:")
    print(results_df)

    return results_df
