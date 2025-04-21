import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import json

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    roc_auc: float
    average_precision: float
    classification_report: str
    timestamp: datetime

class ModelEvaluator:
    def __init__(self):
        self.metrics_dir = "metrics"
        self._setup_metrics_dir()

    def _setup_metrics_dir(self):
        """Create metrics directory if it doesn't exist."""
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # ROC curve and AUC
            roc_auc = 0.0
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            average_precision = 0.0
            if y_prob is not None:
                average_precision = average_precision_score(y_true, y_prob)
            
            # Classification report
            report = classification_report(y_true, y_pred)
            
            # Create metrics object
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                confusion_matrix=conf_matrix,
                roc_auc=roc_auc,
                average_precision=average_precision,
                classification_report=report,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            raise

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {e}")
            raise

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            average_precision = average_precision_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {average_precision:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error plotting precision-recall curve: {e}")
            raise

    def save_metrics(self, metrics: ModelMetrics):
        """Save metrics and plots to files."""
        try:
            timestamp = metrics.timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Save metrics to JSON
            metrics_dict = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "average_precision": metrics.average_precision,
                "timestamp": metrics.timestamp.isoformat(),
                "confusion_matrix": metrics.confusion_matrix.tolist()
            }
            
            metrics_path = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
            with open(metrics_path, "w") as file:
                json.dump(metrics_dict, file, indent=2)
            
            # Save classification report
            report_path = os.path.join(self.metrics_dir, f"classification_report_{timestamp}.txt")
            with open(report_path, "w") as file:
                file.write(metrics.classification_report)
            
            # Save plots
            self.plot_confusion_matrix(
                metrics.confusion_matrix,
                os.path.join(self.metrics_dir, f"confusion_matrix_{timestamp}.png")
            )
            
            logging.info(f"Saved metrics and plots with timestamp: {timestamp}")
            
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
            raise

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Wrapper function for model evaluation."""
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        evaluator.save_metrics(metrics)
        
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "average_precision": metrics.average_precision
        }
        
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.9, 0.1, 0.2, 0.8, 0.3])
    
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")