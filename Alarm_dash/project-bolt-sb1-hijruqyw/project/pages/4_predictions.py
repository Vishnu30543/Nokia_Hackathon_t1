import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

# Page config
st.set_page_config(page_title="Alarm Predictions", page_icon="🔮", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Check if model is available
if 'model' not in st.session_state or st.session_state.model is None:
    st.warning("No trained model available. Please train a model on the ML Modeling page.")
    st.markdown("[← Go to ML Modeling](/pages/3_modeling.py)")
    st.stop()

# Import utility functions
from utils.data_processing import (
    get_next_alarm_prediction_features
)
from models.model_training import (
    predict_next_alarm
)

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">🔮</div>
        <div class="title-container">
            <h1 class="main-title">Alarm Predictions</h1>
            <p class="subtitle">Predict and analyze potential future network alarms</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Make a copy of the data
df = st.session_state.data.copy()

# Convert timestamp to datetime if needed
if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prediction configuration section
st.markdown("## ⚙️ Prediction Configuration")

# Select prediction type
prediction_type = st.radio(
    "What do you want to predict?",
    ["Next Alarm for a Site", "Alarms for All Sites", "Custom Scenario"],
    horizontal=True
)

# Different configuration based on prediction type
if prediction_type == "Next Alarm for a Site":
    # Site selection
    site_options = df['site_id'].unique()
    selected_site = st.selectbox("Select Site", site_options)
    
    # Time selection
    time_options = ["Latest Data", "Custom Time"]
    time_selection = st.radio("Reference Time", time_options, horizontal=True)
    
    if time_selection == "Custom Time":
        # Get min and max dates from data
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        # Date picker
        selected_date = st.date_input("Select Date", max_date, min_value=min_date, max_value=max_date)
        
        # Time picker
        hour = st.slider("Hour", 0, 23, 12)
        minute = st.slider("Minute", 0, 59, 0)
        
        # Create datetime
        prediction_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
    else:
        # Use latest data for the site
        site_data = df[df['site_id'] == selected_site]
        prediction_time = site_data['timestamp'].max() if not site_data.empty else df['timestamp'].max()

elif prediction_type == "Alarms for All Sites":
    # Time selection
    time_options = ["Latest Data", "Custom Time"]
    time_selection = st.radio("Reference Time", time_options, horizontal=True)
    
    if time_selection == "Custom Time":
        # Get min and max dates from data
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        # Date picker
        selected_date = st.date_input("Select Date", max_date, min_value=min_date, max_value=max_date)
        
        # Time picker
        hour = st.slider("Hour", 0, 23, 12)
        minute = st.slider("Minute", 0, 59, 0)
        
        # Create datetime
        prediction_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
    else:
        # Use latest data
        prediction_time = df['timestamp'].max()
    
    # Number of sites to predict
    num_sites = st.slider("Number of Top Sites to Predict", 1, min(10, len(df['site_id'].unique())), 5)
    
    # Get top sites by alarm frequency
    site_counts = df.groupby('site_id').size().sort_values(ascending=False)
    top_sites = site_counts.head(num_sites).index.tolist()

else:  # Custom Scenario
    st.markdown("### Create a Custom Scenario")
    
    # Site selection
    site_options = df['site_id'].unique()
    selected_site = st.selectbox("Select Site", site_options)
    
    # Time selection
    current_date = datetime.now().date()
    selected_date = st.date_input("Scenario Date", current_date)
    hour = st.slider("Hour", 0, 23, 12)
    minute = st.slider("Minute", 0, 59, 0)
    
    # Create datetime
    prediction_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
    
    # Custom parameters
    st.markdown("### Recent Alarm History")
    
    # Previous alarm types
    alarm_types = df['alarm_type'].unique()
    last_alarm = st.selectbox("Last Alarm Type", alarm_types, index=0)
    
    # Severity
    severity_options = df['severity'].unique()
    last_severity = st.selectbox("Last Alarm Severity", severity_options, index=0)
    
    # Temperature
    min_temp = df['temperature'].min() if 'temperature' in df.columns else 18.0
    max_temp = df['temperature'].max() if 'temperature' in df.columns else 35.0
    current_temp = st.slider("Current Temperature (°C)", float(min_temp), float(max_temp), float((min_temp + max_temp) / 2), 0.1)

# Prediction button
if st.button("Generate Prediction"):
    st.markdown("## 🔮 Prediction Results")
    
    # Get model and alarm types mapping
    model = st.session_state.model
    alarm_types_mapping = st.session_state.alarm_types_mapping
    
    # For single site prediction
    if prediction_type == "Next Alarm for a Site":
        # Get features for prediction
        features = get_next_alarm_prediction_features(df, prediction_time, selected_site)
        
        # Make prediction
        prediction = predict_next_alarm(model, features, alarm_types_mapping)
        
        # Display results
        st.markdown(f"### Next Alarm Prediction for Site: {selected_site}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Alarm Type", prediction['alarm_type'])
        
        with col2:
            st.metric("Confidence", f"{prediction['probability']:.2f}")
        
        with col3:
            # Estimated time (random for demo)
            est_time = prediction_time + timedelta(minutes=random.randint(30, 180))
            st.metric("Estimated Time", est_time.strftime("%Y-%m-%d %H:%M"))
        
        # Suggested actions
        st.markdown("### Suggested Preventive Actions")
        
        # Map actions based on alarm type
        action_map = {
            "Link Down": [
                "Check physical connections and cable integrity",
                "Verify power to network devices",
                "Check for configuration changes that might have affected the link"
            ],
            "High CPU": [
                "Check for resource-intensive processes or applications",
                "Review recent traffic patterns for anomalies",
                "Consider load balancing or scaling resources"
            ],
            "Memory Overflow": [
                "Identify and address memory leaks",
                "Increase available memory if possible",
                "Optimize application memory usage"
            ],
            "Power Issue": [
                "Check power supply and connections",
                "Verify UPS functionality",
                "Inspect for power fluctuations"
            ],
            "Temperature Alert": [
                "Check cooling systems",
                "Ensure proper airflow in equipment rooms",
                "Verify temperature sensor functionality"
            ],
            "Packet Loss": [
                "Check for network congestion",
                "Inspect for physical layer issues",
                "Review QoS configurations"
            ],
            "Latency Spike": [
                "Identify bandwidth-heavy applications or users",
                "Check for routing issues",
                "Verify WAN link performance"
            ],
            "Authentication Failure": [
                "Check credential validity",
                "Review authentication server health",
                "Look for potential security incidents"
            ],
            "Config Change": [
                "Review recent configuration changes",
                "Verify configuration compliance",
                "Check for unauthorized changes"
            ],
            "Interface Flapping": [
                "Check for unstable physical connections",
                "Review error counters on interfaces",
                "Look for duplex mismatches"
            ]
        }
        
        # Get actions for the predicted alarm
        predicted_alarm = prediction['alarm_type']
        if predicted_alarm in action_map:
            actions = action_map[predicted_alarm]
            for i, action in enumerate(actions, 1):
                st.markdown(f"{i}. {action}")
        else:
            st.info("No specific actions available for this alarm type.")
    
    # For all sites prediction
    elif prediction_type == "Alarms for All Sites":
        # Container for predictions
        predictions_container = st.container()
        
        with predictions_container:
            st.markdown(f"### Alarm Predictions for Top {len(top_sites)} Sites")
            
            # Create columns for each site
            site_cols = st.columns(len(top_sites))
            
            # Make predictions for each site
            for i, site in enumerate(top_sites):
                # Get features for prediction
                features = get_next_alarm_prediction_features(df, prediction_time, site)
                
                # Make prediction
                prediction = predict_next_alarm(model, features, alarm_types_mapping)
                
                # Determine severity color
                color_map = {
                    "Critical": "red",
                    "Major": "orange",
                    "Minor": "yellow",
                    "Warning": "green",
                    "Info": "blue"
                }
                
                # Get recent alarms for this site
                site_data = df[df['site_id'] == site].sort_values('timestamp', ascending=False)
                recent_severity = site_data['severity'].iloc[0] if not site_data.empty else "Unknown"
                severity_color = color_map.get(recent_severity, "gray")
                
                # Display prediction in card
                with site_cols[i]:
                    st.markdown(f"""
                    <div class="prediction-card" style="border-top: 3px solid {severity_color};">
                        <h4>{site}</h4>
                        <p class="prediction-type">{prediction['alarm_type']}</p>
                        <p class="prediction-prob">Confidence: {prediction['probability']:.2f}</p>
                        <p class="prediction-time">Est. Time: {(prediction_time + timedelta(minutes=random.randint(30, 180))).strftime("%H:%M")}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Hotspot visualization
        st.markdown("### Site Risk Heatmap")
        
        # Import visualization function
        from utils.visualization import plot_hotspot_map
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame()
        
        for site in top_sites:
            # Get features for prediction
            features = get_next_alarm_prediction_features(df, prediction_time, site)
            
            # Make prediction
            prediction = predict_next_alarm(model, features, alarm_types_mapping)
            
            # Get recent data for this site
            site_data = df[df['site_id'] == site].sort_values('timestamp', ascending=False)
            
            if not site_data.empty:
                # Add to dataframe
                site_row = site_data.iloc[0].copy()
                site_row['probability'] = prediction['probability']
                predictions_df = pd.concat([predictions_df, pd.DataFrame([site_row])])
        
        # Plot hotspot map
        if not predictions_df.empty:
            fig = plot_hotspot_map(predictions_df, size_col='probability', color_col='severity_code')
            st.plotly_chart(fig, use_container_width=True)
    
    # For custom scenario
    else:
        # Create a custom feature set
        import random
        
        # Map categorical values to codes
        alarm_type_mapping = {alarm: idx for idx, alarm in enumerate(df['alarm_type'].unique())}
        severity_mapping = {severity: idx for idx, severity in enumerate(df['severity'].unique())}
        
        # Create features dictionary
        custom_features = {
            'last_alarm_type': alarm_type_mapping.get(last_alarm, 0),
            'last_severity': severity_mapping.get(last_severity, 0),
            'last_temperature': current_temp,
            'hour_of_day': prediction_time.hour,
            'day_of_week': prediction_time.weekday(),
            'prev_alarm_1': alarm_type_mapping.get(last_alarm, 0),
            'prev_alarm_2': random.choice(list(alarm_type_mapping.values())),
            'prev_alarm_3': random.choice(list(alarm_type_mapping.values()))
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([custom_features])
        
        # Make prediction
        prediction = predict_next_alarm(model, features_df, alarm_types_mapping)
        
        # Display results
        st.markdown(f"### Custom Scenario Prediction for Site: {selected_site}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Alarm Type", prediction['alarm_type'])
        
        with col2:
            st.metric("Confidence", f"{prediction['probability']:.2f}")
        
        with col3:
            # Severity prediction (simplified)
            severity_options = df['severity'].unique()
            predicted_severity = severity_options[min(len(severity_options)-1, severity_mapping.get(last_severity, 0) + random.choice([-1, 0, 1]))]
            st.metric("Predicted Severity", predicted_severity)
        
        # Scenario analysis
        st.markdown("### Scenario Analysis")
        
        # Create comparison table
        scenario_df = pd.DataFrame([
            {"Parameter": "Site", "Value": selected_site},
            {"Parameter": "Date & Time", "Value": prediction_time.strftime("%Y-%m-%d %H:%M")},
            {"Parameter": "Last Alarm", "Value": last_alarm},
            {"Parameter": "Last Severity", "Value": last_severity},
            {"Parameter": "Temperature", "Value": f"{current_temp}°C"},
            {"Parameter": "Predicted Alarm", "Value": prediction['alarm_type']},
            {"Parameter": "Confidence", "Value": f"{prediction['probability']:.2f}"},
        ])
        
        st.table(scenario_df)
        
        # What-if analysis
        st.markdown("### What-If Analysis")
        st.markdown("How changing parameters affects the prediction:")
        
        # Create tabs for different scenarios
        tab1, tab2, tab3 = st.tabs(["Temperature Change", "Previous Alarm Change", "Time of Day"])
        
        with tab1:
            st.markdown("#### Impact of Temperature Changes")
            
            # Test different temperatures
            temps = [current_temp - 5, current_temp, current_temp + 5]
            temp_predictions = []
            
            for temp in temps:
                # Update features
                custom_features_copy = custom_features.copy()
                custom_features_copy['last_temperature'] = max(min_temp, min(max_temp, temp))
                
                # Make prediction
                features_df = pd.DataFrame([custom_features_copy])
                pred = predict_next_alarm(model, features_df, alarm_types_mapping)
                
                temp_predictions.append({
                    "Temperature": f"{temp:.1f}°C",
                    "Predicted Alarm": pred['alarm_type'],
                    "Confidence": f"{pred['probability']:.2f}"
                })
            
            # Display temperature scenario table
            st.table(pd.DataFrame(temp_predictions))
        
        with tab2:
            st.markdown("#### Impact of Previous Alarm Type")
            
            # Test different previous alarms
            test_alarms = random.sample(list(alarm_types), min(3, len(alarm_types)))
            alarm_predictions = []
            
            for alarm in test_alarms:
                # Update features
                custom_features_copy = custom_features.copy()
                custom_features_copy['prev_alarm_1'] = alarm_type_mapping.get(alarm, 0)
                
                # Make prediction
                features_df = pd.DataFrame([custom_features_copy])
                pred = predict_next_alarm(model, features_df, alarm_types_mapping)
                
                alarm_predictions.append({
                    "Previous Alarm": alarm,
                    "Predicted Alarm": pred['alarm_type'],
                    "Confidence": f"{pred['probability']:.2f}"
                })
            
            # Display alarm scenario table
            st.table(pd.DataFrame(alarm_predictions))
        
        with tab3:
            st.markdown("#### Impact of Time of Day")
            
            # Test different times
            hours = [8, 12, 18, 22]
            time_predictions = []
            
            for hour in hours:
                # Update features
                custom_features_copy = custom_features.copy()
                custom_features_copy['hour_of_day'] = hour
                
                # Make prediction
                features_df = pd.DataFrame([custom_features_copy])
                pred = predict_next_alarm(model, features_df, alarm_types_mapping)
                
                time_predictions.append({
                    "Time": f"{hour}:00",
                    "Predicted Alarm": pred['alarm_type'],
                    "Confidence": f"{pred['probability']:.2f}"
                })
            
            # Display time scenario table
            st.table(pd.DataFrame(time_predictions))

# Add import for random (used in predictions)
import random

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to ML Modeling](/pages/3_modeling.py)")

with col2:
    st.markdown("[Continue to Visualizations →](/pages/5_visualizations.py)")