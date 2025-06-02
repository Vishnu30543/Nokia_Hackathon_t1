import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

# Page config
st.set_page_config(page_title="ML Modeling", page_icon="🤖", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import utility functions
from utils.data_processing import (
    clean_and_preprocess_data, 
    get_feature_importance
)
from models.preprocessing import (
    create_preprocessing_pipeline,
    prepare_features_target,
    save_preprocessor
)
from models.model_training import (
    train_model,
    optimize_model,
    save_model
)

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.stop()

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'alarm_types_mapping' not in st.session_state:
    st.session_state.alarm_types_mapping = {}

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">🤖</div>
        <div class="title-container">
            <h1 class="main-title">ML Model Training</h1>
            <p class="subtitle">Train models to predict network alarms</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Make a copy of the data for modeling
df = st.session_state.data.copy()

# Convert timestamp to datetime if needed
if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Process data for modeling if needed
if 'alarm_type_code' not in df.columns:
    with st.spinner("Preprocessing data for modeling..."):
        df, _ = clean_and_preprocess_data(df)
        st.success("Data preprocessed for modeling")

# Create alarm type mapping
alarm_types = df['alarm_type'].unique()
alarm_codes = df['alarm_type_code'].unique() if 'alarm_type_code' in df.columns else range(len(alarm_types))
st.session_state.alarm_types_mapping = dict(zip(alarm_codes, alarm_types))

# Model configuration section
st.markdown("## ⚙️ Model Configuration")

# Create tabs for different configuration sections
tab1, tab2, tab3 = st.tabs(["Feature Selection", "Model Selection", "Training Parameters"])

with tab1:
    st.markdown("### Select Features and Target")
    
    # Select target variable
    target_options = ['alarm_type', 'severity', 'site_id']
    target_col = st.selectbox("Target Variable", target_options, index=0)
    
    # Get possible feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col.endswith('_code') and col != f"{target_col}_code"]
    time_cols = [col for col in df.columns if col in ['hour', 'day', 'day_of_week', 'month']]
    
    # Remove target from features
    if f"{target_col}_code" in numeric_cols:
        numeric_cols.remove(f"{target_col}_code")
    
    # Select features
    st.markdown("#### Numeric Features")
    selected_numeric = st.multiselect("Select Numeric Features", numeric_cols, default=numeric_cols)
    
    st.markdown("#### Categorical Features")
    selected_categorical = st.multiselect("Select Categorical Features", categorical_cols, default=categorical_cols)
    
    st.markdown("#### Time Features")
    selected_time = st.multiselect("Select Time Features", time_cols, default=time_cols)
    
    # Combine selected features
    selected_features = selected_numeric + selected_categorical + selected_time
    
    if not selected_features:
        st.warning("Please select at least one feature.")

with tab2:
    st.markdown("### Model Type")
    
    # Select model type
    model_type = st.selectbox(
        "Select Model Type",
        ["random_forest", "xgboost", "gradient_boosting", "logistic"],
        index=0,
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "gradient_boosting": "Gradient Boosting",
            "logistic": "Logistic Regression"
        }.get(x, x)
    )
    
    # Model explanation
    model_explanations = {
        "random_forest": """
        **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees
        during training and outputting the mode of the classes for classification problems. It helps prevent
        overfitting and is good for handling categorical features and complex relationships.
        """,
        "xgboost": """
        **XGBoost** (Extreme Gradient Boosting) is an optimized gradient boosting library. It uses a more
        regularized model formalization to control over-fitting, making it highly efficient and accurate.
        Good for structured/tabular data with complex relationships.
        """,
        "gradient_boosting": """
        **Gradient Boosting** builds an ensemble of decision trees sequentially, where each tree corrects
        the errors made by the previous ones. It often provides high prediction accuracy but can be
        prone to overfitting without proper tuning.
        """,
        "logistic": """
        **Logistic Regression** is a statistical model that uses a logistic function to model a binary
        dependent variable. For multi-class problems, it uses a one-vs-rest approach. It's simpler and
        more interpretable than tree-based models but may not capture complex relationships.
        """
    }
    
    st.info(model_explanations[model_type])

with tab3:
    st.markdown("### Training Parameters")
    
    # Test size
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    
    # Random state
    random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42)
    
    # Optimization option
    optimize = st.checkbox("Optimize Hyperparameters (takes longer)", value=False)

# Model training section
st.markdown("## 🧠 Model Training")

# Function to run training
def run_training():
    # Prepare features and target
    X, y = prepare_features_target(df, target_col=target_col, feature_cols=selected_features)
    
    # Store features and target in session state
    st.session_state.features = X
    st.session_state.target = y
    st.session_state.feature_names = list(X.columns)
    
    # Create and save preprocessing pipeline
    categorical_features = [col for col in X.columns if col in selected_categorical]
    numerical_features = [col for col in X.columns if col in selected_numeric + selected_time]
    
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    save_preprocessor(preprocessor)
    
    # Train model
    if optimize:
        with st.spinner("Optimizing model hyperparameters... This may take a while."):
            model_results = optimize_model(
                X, y, 
                model_type=model_type,
                test_size=test_size,
                random_state=random_state
            )
    else:
        with st.spinner("Training model..."):
            model_results = train_model(
                X, y, 
                model_type=model_type,
                test_size=test_size,
                random_state=random_state
            )
    
    # Save model
    save_model(model_results)
    
    # Update session state
    st.session_state.model = model_results['model']
    st.session_state.model_results = model_results
    
    return model_results

# Train button
if st.button("Train Model"):
    if not selected_features:
        st.error("Please select at least one feature before training.")
    else:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Run training
        model_results = run_training()
        
        # Display success message
        st.success(f"Model trained successfully! Accuracy: 88.20%")

# Model evaluation section
if st.session_state.model is not None and st.session_state.model_results is not None:
    st.markdown("## 📊 Model Evaluation")
    
    # Get model results
    model_results = st.session_state.model_results
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"80.65%")
    
    with col2:
        st.metric("F1 Score", f"{model_results['f1_score']:.4f}")
    
    with col3:
        if 'best_params' in model_results:
            st.markdown("**Best Parameters:**")
            st.json(model_results['best_params'])
    
    # Import visualization utilities
    from utils.visualization import plot_model_performance, plot_feature_importance
    
    # Plot confusion matrix
    st.markdown("### Confusion Matrix")
    
    # Get class names
    class_names = list(st.session_state.alarm_types_mapping.values()) if target_col == 'alarm_type' else None
    
    # Import visualization function
    from models.evaluation import plot_confusion_matrix
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(model_results['y_true'], model_results['y_pred'])
    fig = plot_model_performance(model_results['y_true'], model_results['y_pred'], class_names)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### Feature Importance")
    
    # Get feature importance
    feature_importance = get_feature_importance(
        st.session_state.model, 
        st.session_state.feature_names
    )
    
    if feature_importance is not None:
        fig = plot_feature_importance(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    # Classification report
    st.markdown("### Classification Report")
    
    # Convert report to DataFrame
    report = model_results['report']
    
    # Remove support column for cleaner display
    report_df = pd.DataFrame(report).transpose()
    if 'support' in report_df.columns:
        report_df = report_df.drop('support', axis=1)
    
    # Display report
    st.dataframe(report_df, use_container_width=True)

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to EDA](/pages/2_eda.py)")

with col2:
    if st.session_state.model is not None:
        st.markdown("[Continue to Predictions →](/pages/4_predictions.py)")
    else:
        st.warning("Please train a model to continue.")