import pickle
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TRAINED_MODEL_FILE = os.getenv("TRAINED_MODEL_FILE")

def load_model():
    with open(TRAINED_MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    return model

def generate_predictions():
    model = load_model()
    with open("market_trends.json", "r") as file:
        market_trends = json.load(file)
    trends_df = pd.DataFrame([market_trends])

    predictions = model.predict(trends_df)
    confidence_scores = model.predict_proba(trends_df)[:, 1]

    return predictions, confidence_scores

if __name__ == "__main__":
    predictions, confidence_scores = generate_predictions()
    for prediction, confidence in zip(predictions, confidence_scores):
        print(f"Recommendation: {prediction}, Confidence Score: {confidence}")