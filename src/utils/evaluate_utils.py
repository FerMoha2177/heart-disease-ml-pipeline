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


# load model
def load_trained_models():
    """Load all trained models from disk"""
    models = {}
    model_dir = "../models"
    
    for filename in os.listdir(model_dir):
        if filename.endswith("_tuned.joblib"):
            model_name = filename.replace("_tuned.joblib", "")
            model_path = os.path.join(model_dir, filename)
            
            try:
                model = load_model(model_path)
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    return models

# Evaluation
def evaluate_model(model, X_test, y_test, model_name):

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