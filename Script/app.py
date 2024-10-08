import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Define the path to your saved model here
MODEL_PATH = "random_forest_model.pkl"  # Update this with your actual model path

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
        feature_names = set(model.feature_names_in_)  # Use set for intersection
    else:
        feature_names = set(model.get_feature_names_out())

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
