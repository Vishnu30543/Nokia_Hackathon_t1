import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import joblib
import os
import streamlit as st

def train_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    """
    Train a machine learning model
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Features for model training
    y : Series or array-like
        Target variable for model training
    model_type : str
        Type of model to train ('random_forest', 'xgboost', 'gradient_boosting', 'logistic')
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing the trained model, metrics, and other information
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        f1 = f1_score(y_test, y_pred, average='weighted')
    except:
        f1 = 0
    
    # Create report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Return results
    return {
        'model': model,
        'model_type': model_type,
        'accuracy': accuracy,
        'f1_score': f1,
        'report': report,
        'y_true': y_test,
        'y_pred': y_pred
    }

def optimize_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    """
    Optimize model hyperparameters using grid search
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Features for model training
    y : Series or array-like
        Target variable for model training
    model_type : str
        Type of model to train ('random_forest', 'xgboost', 'gradient_boosting', 'logistic')
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing the optimized model, metrics, and other information
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize model and parameter grid based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=random_state)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga']
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set up grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Run grid search with progress updates
    with st.spinner('Optimizing model hyperparameters...'):
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Update progress
        progress_bar.progress(100)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        f1 = f1_score(y_test, y_pred, average='weighted')
    except:
        f1 = 0
    
    # Create report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Return results
    return {
        'model': best_model,
        'model_type': model_type,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'f1_score': f1,
        'report': report,
        'y_true': y_test,
        'y_pred': y_pred
    }

def save_model(model_results, model_dir='models'):
    """
    Save the trained model and results to disk
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing the model and results
    model_dir : str
        Directory to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract model
    model = model_results['model']
    model_type = model_results['model_type']
    
    # Save model
    joblib.dump(model, os.path.join(model_dir, f'{model_type}_model.joblib'))
    
    # Save results (excluding model object)
    results_copy = model_results.copy()
    results_copy.pop('model')
    joblib.dump(results_copy, os.path.join(model_dir, f'{model_type}_results.joblib'))

def load_model(model_type='random_forest', model_dir='models'):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_type : str
        Type of model to load
    model_dir : str
        Directory where the model is saved
        
    Returns:
    --------
    tuple
        Loaded model and results
    """
    # Check if model exists
    model_path = os.path.join(model_dir, f'{model_type}_model.joblib')
    results_path = os.path.join(model_dir, f'{model_type}_results.joblib')
    
    if not (os.path.exists(model_path) and os.path.exists(results_path)):
        return None, None
    
    # Load model and results
    model = joblib.load(model_path)
    results = joblib.load(results_path)
    
    # Add model to results
    results['model'] = model
    
    return model, results

def predict_next_alarm(model, features, alarm_types):
    """
    Predict the next alarm type
    
    Parameters:
    -----------
    model : trained model
        Trained classification model
    features : DataFrame
        Features for prediction
    alarm_types : list or dict
        Mapping from alarm code to alarm type
        
    Returns:
    --------
    dict
        Prediction results
    """
    # Make prediction
    try:
        # Get predicted class
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            max_prob = probabilities.max()
        else:
            # For models without predict_proba
            max_prob = 1.0
        
        # Map prediction to alarm type
        if isinstance(alarm_types, dict):
            # If alarm_types is a dictionary
            alarm_type = alarm_types.get(prediction, f"Unknown ({prediction})")
        else:
            # If alarm_types is a list
            try:
                alarm_type = alarm_types[prediction]
            except (IndexError, TypeError):
                alarm_type = f"Unknown ({prediction})"
        
        return {
            'prediction': prediction,
            'alarm_type': alarm_type,
            'probability': max_prob
        }
    
    except Exception as e:
        # Return error information
        return {
            'error': str(e),
            'prediction': None,
            'alarm_type': 'Error in prediction',
            'probability': 0.0
        }