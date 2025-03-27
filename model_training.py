import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HISTORICAL_DATA_FILE = os.getenv("HISTORICAL_DATA_FILE")
TRAINED_MODEL_FILE = os.getenv("TRAINED_MODEL_FILE")

def load_training_data():
    with open(HISTORICAL_DATA_FILE, "r") as file:
        historical_data = json.load(file)
    return pd.DataFrame(historical_data)

def train_model():
    data = load_training_data()
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return model

if __name__ == "__main__":
    model = train_model()
    with open(TRAINED_MODEL_FILE, "wb") as file:
        pickle.dump(model, file)