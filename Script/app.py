import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Define the path to your saved model here
MODEL_PATH = "random_forest_model.pkl"  

# Define the data model for incoming test data
class TestData(BaseModel):
    CustomerId: str
    # Define other fields that your model expects
    # Example: Feature1: float, Feature2: float, ...

def calculate_credit_score(probability: float) -> float:
    Score_min = 300
    Score_max = 900
    score = Score_min + (1 - probability) * (Score_max - Score_min)
    return score

def predict_probabilities(test_data: pd.DataFrame) -> pd.DataFrame:
    # Load the saved model
    model = joblib.load(MODEL_PATH)

    # Get the feature names from the model
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
        'Prob_Good_Credit': predicted_probabilities[:, 1]  # Probability of class 1
    })

    # Calculate credit scores based on the probabilities
    results_df['Credit_Score'] = results_df['Prob_Good_Credit'].apply(calculate_credit_score)

    return results_df

@app.post("/predict")
def predict_credit_scores(test_data: List[TestData]):
    """
    Endpoint to predict credit scores based on provided test data.

    Parameters:
    - test_data: List[TestData], the input data for prediction.

    Returns:
    - List[dict]: Predicted probabilities and credit scores.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict() for data in test_data])
    
    try:
        results_df = predict_probabilities(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert results to a list of dictionaries for the response
    results = results_df.to_dict(orient="records")

    return {"predictions": results}

@app.post("/upload")
async def upload_test_data(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV file containing test data.

    Parameters:
    - file: UploadFile, the CSV file containing the test data.

    Returns:
    - List[dict]: Predicted probabilities and credit scores.
    """
    # Read the uploaded CSV file into a DataFrame
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
    
    # Ensure CustomerId column exists
    if 'CustomerId' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'CustomerId' column.")

    try:
        results_df = predict_probabilities(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert results to a list of dictionaries for the response
    results = results_df.to_dict(orient="records")

    return {"predictions": results}
