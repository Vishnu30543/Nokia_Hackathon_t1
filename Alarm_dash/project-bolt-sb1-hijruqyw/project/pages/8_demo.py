import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Page config
st.set_page_config(page_title="Demo Simulation", page_icon="🎮", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import utility functions
from utils.simulation import create_simulation_widget
from utils.data_processing import load_sample_data
from models.model_training import load_model

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">🎮</div>
        <div class="title-container">
            <h1 class="main-title">Demo Simulation</h1>
            <p class="subtitle">See the prediction system in action with simulated network alarms</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Check if data is available, otherwise load sample data
if 'data' not in st.session_state or st.session_state.data is None:
    st.session_state.data = load_sample_data()

# Ensure data is properly processed
if 'alarm_type_code' not in st.session_state.data.columns:
    from utils.data_processing import clean_and_preprocess_data
    st.session_state.data, _ = clean_and_preprocess_data(st.session_state.data)

# Make a copy of the data
df = st.session_state.data.copy()

# Convert timestamp to datetime if needed
if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create alarm type mapping
alarm_types = df['alarm_type'].unique()
alarm_codes = df['alarm_type_code'].unique() if 'alarm_type_code' in df.columns else range(len(alarm_types))
alarm_types_mapping = dict(zip(alarm_codes, alarm_types))

# Check if model exists, otherwise train a simple model
if 'model' not in st.session_state or st.session_state.model is None:
    # Try to load a saved model
    model_dir = "models"
    model, results = load_model(model_type="random_forest", model_dir=model_dir)
    
    if model is not None:
        st.session_state.model = model
    else:
        # Train a simple model
        from models.preprocessing import prepare_features_target
        from models.model_training import train_model
        
        with st.spinner("No trained model found. Training a simple model for simulation..."):
            # Select features for a simple model
            if 'hour' not in df.columns:
                df['hour'] = df['timestamp'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Prepare features and target
            feature_cols = ['site_id_code', 'severity_code', 'hour', 'day_of_week']
            if 'temperature' in df.columns:
                feature_cols.append('temperature')
            
            X, y = prepare_features_target(df, target_col='alarm_type', feature_cols=feature_cols)
            
            # Train a simple model
            model_results = train_model(X, y, model_type="random_forest", test_size=0.2, random_state=42)
            
            # Save model
            from models.model_training import save_model
            save_model(model_results, model_dir=model_dir)
            
            # Update session state
            st.session_state.model = model_results['model']
            
            st.success("Simple model trained for simulation purposes.")

# Simulation options
st.markdown("## ⚙️ Simulation Configuration")

# Create columns for simulation options
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Simulation Options")
    
    # Simulation type
    sim_type = st.radio(
        "Simulation Type",
        ["Real-time Alarm Simulation", "Scenario-based Simulation"],
        index=0
    )
    
    if sim_type == "Real-time Alarm Simulation":
        # Real-time simulation options
        st.markdown("#### Real-time Simulation Settings")
        
        # Number of sites
        num_sites = st.slider("Number of Sites", 1, 10, 5)
        
        # Time factor (speed up)
        time_factor = st.slider("Time Acceleration Factor", 1, 50, 20)
        
        # Duration
        duration = st.slider("Simulation Duration (minutes)", 1, 10, 5)
        
    else:
        # Scenario-based simulation options
        st.markdown("#### Scenario Settings")
        
        # Scenario selection
        scenario = st.selectbox(
            "Select Scenario",
            [
                "Network Outage",
                "Temperature Spike",
                "Maintenance Window",
                "DDoS Attack",
                "Power Fluctuation"
            ]
        )
        
        # Site selection
        sites = df['site_id'].unique()
        selected_sites = st.multiselect("Select Sites", sites, default=sites[:3] if len(sites) >= 3 else sites)
        
        # Duration
        duration = st.slider("Scenario Duration (minutes)", 1, 10, 5)

with col2:
    st.markdown("### Visualization Options")
    
    # Display options
    st.markdown("#### Display Settings")
    
    # Show probability threshold
    prob_threshold = st.slider("Probability Threshold for Alerts", 0.0, 1.0, 0.5)
    
    # Visualization types
    show_timeline = st.checkbox("Show Prediction Timeline", value=True)
    show_map = st.checkbox("Show Network Map", value=True)
    show_metrics = st.checkbox("Show Performance Metrics", value=True)

# Simulation controls
st.markdown("## 🎮 Simulation Controls")

# Create simulation widget
if sim_type == "Real-time Alarm Simulation":
    # Use the simulation widget from utils
    create_simulation_widget(df, st.session_state.model, alarm_types_mapping)
    
else:  # Scenario-based simulation
    # Create a custom scenario simulation
    
    # Define scenario parameters
    scenario_params = {
        "Network Outage": {
            "description": "Simulates a network outage affecting multiple sites, starting with Link Down alarms followed by cascading effects.",
            "alarm_sequence": ["Link Down", "Packet Loss", "Latency Spike"],
            "severity_sequence": ["Critical", "Major", "Major"],
            "affected_metric": "connectivity"
        },
        "Temperature Spike": {
            "description": "Simulates a temperature increase in data centers, leading to temperature alerts and potential hardware issues.",
            "alarm_sequence": ["Temperature Alert", "High CPU", "Memory Overflow"],
            "severity_sequence": ["Warning", "Major", "Critical"],
            "affected_metric": "temperature"
        },
        "Maintenance Window": {
            "description": "Simulates planned maintenance activities with configuration changes and temporary service impacts.",
            "alarm_sequence": ["Config Change", "Link Down", "Authentication Failure", "Config Change"],
            "severity_sequence": ["Info", "Minor", "Warning", "Info"],
            "affected_metric": "availability"
        },
        "DDoS Attack": {
            "description": "Simulates a distributed denial of service attack with traffic spikes and performance degradation.",
            "alarm_sequence": ["Latency Spike", "High CPU", "Packet Loss", "Memory Overflow"],
            "severity_sequence": ["Warning", "Major", "Critical", "Critical"],
            "affected_metric": "traffic"
        },
        "Power Fluctuation": {
            "description": "Simulates power supply issues causing device reboots and connectivity problems.",
            "alarm_sequence": ["Power Issue", "Link Down", "Interface Flapping", "Power Issue"],
            "severity_sequence": ["Major", "Critical", "Major", "Minor"],
            "affected_metric": "power"
        }
    }
    
    # Get current scenario
    current_scenario = scenario_params[scenario]
    
    # Display scenario description
    st.info(current_scenario["description"])
    
    # Scenario simulation function
    def run_scenario_simulation():
        # Container for simulation output
        output_container = st.empty()
        timeline_container = st.empty()
        metrics_container = st.empty()
        map_container = st.empty()
        
        # Initialize simulation data
        all_alarms = []
        all_predictions = []
        
        # Set up metrics
        correct_predictions = 0
        total_predictions = 0
        avg_prediction_time = 0
        
        # Get scenario parameters
        alarm_sequence = current_scenario["alarm_sequence"]
        severity_sequence = current_scenario["severity_sequence"]
        affected_metric = current_scenario["affected_metric"]
        
        # Start time
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration)
        
        # Update function
        def update_ui():
            with output_container.container():
                # Display latest alarms
                st.subheader("Latest Network Alarms")
                
                # Create columns for latest alarms
                if all_alarms:
                    alarm_cols = st.columns(min(3, len(all_alarms)))
                    
                    for i, alarm in enumerate(all_alarms[-3:]):
                        with alarm_cols[i % 3]:
                            # Determine severity color
                            severity_color = {
                                "Critical": "#FF5252",
                                "Major": "#FF9800",
                                "Minor": "#FFC107",
                                "Warning": "#8BC34A",
                                "Info": "#2196F3"
                            }.get(alarm['severity'], "#9E9E9E")
                            
                            # Display alarm card
                            st.markdown(f"""
                            <div class="alarm-card" style="border-left: 4px solid {severity_color};">
                                <div class="alarm-header">
                                    <span class="alarm-type">{alarm['alarm_type']}</span>
                                    <span class="alarm-time">{alarm['timestamp'].strftime('%H:%M:%S')}</span>
                                </div>
                                <div class="alarm-details">
                                    <p><strong>Site:</strong> {alarm['site_id']}</p>
                                    <p><strong>Severity:</strong> {alarm['severity']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Update timeline
            if show_timeline and all_alarms:
                with timeline_container.container():
                    st.subheader("Alarm Prediction Timeline")
                    
                    # Import visualization function
                    from utils.visualization import plot_prediction_timeline
                    
                    # Convert alarms to format for timeline
                    timeline_predictions = [
                        {
                            'timestamp': p['timestamp'],
                            'alarm_type': p['alarm_type'],
                            'probability': p['probability'],
                            'color': {'Critical': 'red', 'Major': 'orange', 'Minor': 'yellow', 
                                    'Warning': 'green', 'Info': 'blue'}.get(p['severity'], 'gray')
                        }
                        for p in all_predictions
                    ]
                    
                    timeline_actuals = [
                        {
                            'timestamp': a['timestamp'],
                            'alarm_type': a['alarm_type'],
                            'color': {'Critical': 'red', 'Major': 'orange', 'Minor': 'yellow', 
                                    'Warning': 'green', 'Info': 'blue'}.get(a['severity'], 'gray')
                        }
                        for a in all_alarms
                    ]
                    
                    # Plot timeline
                    fig = plot_prediction_timeline(timeline_predictions, timeline_actuals)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Update metrics
            if show_metrics and all_alarms:
                with metrics_container.container():
                    st.subheader("Simulation Metrics")
                    
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Alarms Generated", len(all_alarms))
                    
                    with metric_cols[1]:
                        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0
                        st.metric("Prediction Accuracy", f"{accuracy:.2f}")
                    
                    with metric_cols[2]:
                        avg_prob = sum(p['probability'] for p in all_predictions) / len(all_predictions) if all_predictions else 0
                        st.metric("Avg. Confidence", f"{avg_prob:.2f}")
                    
                    with metric_cols[3]:
                        st.metric("Avg. Prediction Time", f"{avg_prediction_time:.1f}s")
            
            # Update network map
            if show_map and all_alarms:
                with map_container.container():
                    st.subheader("Network Alarm Map")
                    
                    # Import visualization function
                    from utils.visualization import plot_hotspot_map
                    
                    # Create data for map
                    map_data = pd.DataFrame(all_alarms)
                    
                    # Add severity code for visualization
                    severity_map = {
                        "Critical": 4,
                        "Major": 3,
                        "Minor": 2,
                        "Warning": 1,
                        "Info": 0
                    }
                    map_data['severity_code'] = map_data['severity'].map(severity_map)
                    
                    # Plot map
                    fig = plot_hotspot_map(map_data)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Run simulation if button is pressed
        if st.button("Start Scenario Simulation"):
            # Initialize progress bar
            progress = st.progress(0)
            
            # Run simulation
            with st.spinner(f"Running {scenario} simulation..."):
                # Loop through simulation time
                current_time = start_time
                step = 0
                
                while current_time < end_time:
                    # Calculate progress
                    elapsed = (current_time - start_time).total_seconds()
                    total_duration = (end_time - start_time).total_seconds()
                    progress_value = min(1.0, elapsed / total_duration)
                    progress.progress(progress_value)
                    
                    # Generate alarm based on scenario
                    if step < len(alarm_sequence):
                        # Select a site
                        site = random.choice(selected_sites)
                        
                        # Create alarm
                        alarm = {
                            'timestamp': current_time,
                            'alarm_type': alarm_sequence[step],
                            'site_id': site,
                            'severity': severity_sequence[step],
                            'temperature': round(random.uniform(18.0, 35.0), 1),
                            'duration_minutes': random.randint(1, 180)
                        }
                        
                        # Add to alarms list
                        all_alarms.append(alarm)
                        
                        # Generate prediction
                        from utils.data_processing import get_next_alarm_prediction_features
                        from models.model_training import predict_next_alarm
                        
                        # Make prediction for next alarm
                        next_step = (step + 1) % len(alarm_sequence)
                        features = get_next_alarm_prediction_features(df, current_time, site)
                        prediction = predict_next_alarm(st.session_state.model, features, alarm_types_mapping)
                        
                        # Add time component to prediction
                        pred_time = current_time + timedelta(seconds=random.randint(30, 120))
                        predicted_alarm = {
                            'timestamp': pred_time,
                            'alarm_type': prediction['alarm_type'],
                            'site_id': site,
                            'severity': severity_sequence[next_step] if next_step < len(severity_sequence) else "Unknown",
                            'temperature': alarm['temperature'],
                            'duration_minutes': alarm['duration_minutes'],
                            'probability': prediction['probability']
                        }
                        
                        # Add to predictions list
                        all_predictions.append(predicted_alarm)
                        
                        # Update metrics
                        total_predictions += 1
                        if predicted_alarm['alarm_type'] == alarm_sequence[next_step]:
                            correct_predictions += 1
                        
                        avg_prediction_time = random.uniform(0.8, 3.5)  # Simulated prediction time
                        
                        # Update UI
                        update_ui()
                        
                        # Increment step
                        step += 1
                    
                    # Advance time
                    wait_time = random.randint(5, 15) / time_factor
                    time.sleep(wait_time)
                    current_time = current_time + timedelta(seconds=random.randint(15, 30))
            
            # Final update
            update_ui()
            
            # Complete progress
            progress.progress(1.0)
            
            st.success("Simulation completed!")
    
    # Run the scenario simulation
    run_scenario_simulation()

# Summary and explanation
st.markdown("## 📝 How the Simulation Works")

st.markdown("""
This simulation demonstrates how the Network Alarm Prediction System processes alarm data and
generates predictions for future alarms. Here's what's happening behind the scenes:

1. **Data Collection**: In a real system, data would be collected from network devices through monitoring
   systems. In this simulation, we're generating synthetic alarm data based on realistic patterns.

2. **Feature Engineering**: The system extracts relevant features from the alarm data, including:
   - Previous alarm types and their patterns
   - Site-specific characteristics
   - Time-based features (hour of day, day of week)
   - Environmental conditions (temperature)

3. **Prediction Generation**: The trained machine learning model analyzes these features to predict:
   - The next likely alarm type
   - Probability/confidence of the prediction
   - Estimated time window for the alarm

4. **Visualization**: The system displays the predictions alongside actual alarms to show how
   well the model is performing in real-time.

This simulation is a simplified version of how the full system would operate in a production
environment, where it would be integrated with actual network monitoring systems and would
process real alarm data.
""")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Architecture](/pages/7_architecture.py)")

with col2:
    st.markdown("[Return to Home](/?page=)")