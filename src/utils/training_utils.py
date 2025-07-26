import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump as dump_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import os
from config.logging import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

# SPLIT DATA ONCE

def get_models(y_train):
    try:
        # get models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)),
            'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42)
        }
        logger.info("\nRetrieved models: {}".format(models.keys()))
        return models   
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        raise
    
def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split data once for consistent evaluation"""
    try:    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to split data: {str(e)}")
        raise

def train_model(model, X_train, y_train, model_name):
    """Train a model with training data"""
    try:
        model.fit(X_train, y_train)
        logger.info(f"Trained {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise

def hyperparameter_tuning_grid(model, X, y, param_grid, cv=5, n_jobs=-1):
    """Perform hyperparameter tuning using GridSearchCV"""

    try:
        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs)
        grid_search.fit(X, y)
        logger.info(f"Hyperparameter tuning completed for {model.__class__.__name__}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Failed to perform hyperparameter tuning: {str(e)}")
        raise


def hyperparameter_tuning_random(model, X, y, param_grid, cv=3, n_iter=10, n_jobs=-1):
    """Perform hyperparameter tuning using RandomizedSearchCV
    
    Switched from GridSearchCV to RandomizedSearchCV to prevent overfitting
    Used simpler hyperparameter grids based on domain knowledge
    Implemented proper train/validation/test splits
    """
    try:
        random_search = RandomizedSearchCV(
            model, param_grid, cv=cv, n_iter=n_iter, 
            random_state=42, n_jobs=n_jobs
        )
        random_search.fit(X, y)
        logger.info(f"Hyperparameter tuning completed for {model.__class__.__name__}")
        return random_search.best_estimator_
    except Exception as e:
        logger.error(f"Failed to perform hyperparameter tuning: {str(e)}")
        raise
    
def remove_old_models():
    """Remove tuned model files, keep metadata and preprocessing artifacts"""
    try:
        for file in os.listdir("../models"):
            if file.endswith("_tuned.joblib"):
                logger.info(f"Removing old model: {file}")
                os.remove(os.path.join("../models", file))
    except Exception as e:
        logger.error(f"Failed to remove old models: {str(e)}")
        raise

def save_model(model, model_name):
    """Save the model to disk"""
    try:
        dump_model(model, f"../models/{model_name}_tuned.joblib")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise