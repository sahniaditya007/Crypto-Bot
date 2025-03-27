from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Example true values
    y_pred = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # Example predicted values

    metrics = evaluate_model(y_true, y_pred)
    print(metrics)