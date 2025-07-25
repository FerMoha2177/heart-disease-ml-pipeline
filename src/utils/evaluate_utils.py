import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.utils.data_utils import drop_id
from joblib import load as load_model
import os
from config.logging import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

MODEL_DIR = "../models"


# load model
def load_trained_models():
    """Load all trained models from disk"""
    models = {}
    
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith("_tuned.joblib"):
            model_name = filename.replace("_tuned.joblib", "")
            model_path = os.path.join(MODEL_DIR, filename)
            
            try:
                model = load_model(model_path)
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    return models

# Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model on test data

    This function evaluates a model on test data and returns the evaluation results and Calculate the metrics.

    - Accuracy: The ratio of correct predictions to total predictions. ( Misleading with imbalanced datasets. )
    - Precision: The ratio of true positive predictions to total positive predictions. (Of all predicted positives, how many were actually correct.)
    - Recall: The ratio of true positive predictions to total actual positive cases. (Of all actual positives, how many were predicted correctly?)
    - F1-Score: The harmonic mean of precision and recall. (Balances precision and recall. AKA the Harmonic Mean of precision and recall.)
    - ROC-AUC (Receiver Operating Characteristic Area Under Curve): The area under the receiver operating characteristic curve. (Measures the ability of the model to distinguish between positive and negative cases.)
    
    Args:
        model (sklearn.base.BaseEstimator): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_name (str): Name of the model
    
    Returns:
        dict: Evaluation results
    """

    try:
        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
        # Calculate metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'model': model
        }
    
        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")
    
        return results
    
    except Exception as e:
        logger.error(f"Failed to evaluate model: {str(e)}")
        raise

def clear_worse_models(best_model_name):
    """Remove models that are worse than the best model"""
    try:
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".joblib"):
                if file != best_model_name:
                    logger.info(f"Removing worse model: {file}")
                    os.remove(os.path.join(MODEL_DIR, file))
    except Exception as e:
        logger.error(f"Failed to clear worse models: {str(e)}")
        raise