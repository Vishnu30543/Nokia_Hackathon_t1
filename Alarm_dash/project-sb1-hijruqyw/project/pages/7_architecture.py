import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(page_title="System Architecture", page_icon="🏗️", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">🏗️</div>
        <div class="title-container">
            <h1 class="main-title">System Architecture</h1>
            <p class="subtitle">Design and implementation details of the alarm prediction system</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Architecture overview
st.markdown("## 🏗️ System Architecture Overview")

st.markdown("""
This page details the architecture and implementation of the Network Alarm Prediction System,
including data flow, model deployment, and integration with network monitoring systems.
""")

# Architecture diagram
st.markdown("### System Architecture Diagram")

# Create architecture diagram with Plotly
fig = go.Figure()

# Define node positions
nodes = {
    "network_devices": {"x": 0, "y": 0, "name": "Network Devices", "size": 30, "color": "#2CA58D"},
    "monitoring": {"x": 1, "y": 0, "name": "Monitoring System", "size": 30, "color": "#2CA58D"},
    "data_ingest": {"x": 2, "y": 0, "name": "Data Ingestion", "size": 30, "color": "#0A2342"},
    "storage": {"x": 3, "y": 0, "name": "Data Storage", "size": 30, "color": "#0A2342"},
    "preprocessing": {"x": 2, "y": -1, "name": "Data Preprocessing", "size": 30, "color": "#0A2342"},
    "feature_eng": {"x": 3, "y": -1, "name": "Feature Engineering", "size": 30, "color": "#0A2342"},
    "model_training": {"x": 4, "y": -1, "name": "Model Training", "size": 30, "color": "#F3A712"},
    "model_eval": {"x": 5, "y": -1, "name": "Model Evaluation", "size": 30, "color": "#F3A712"},
    "prediction": {"x": 4, "y": 0, "name": "Prediction Engine", "size": 30, "color": "#F3A712"},
    "api": {"x": 5, "y": 0, "name": "API Service", "size": 30, "color": "#E6212B"},
    "visualization": {"x": 6, "y": 0, "name": "Visualization", "size": 30, "color": "#E6212B"},
    "alerts": {"x": 6, "y": -1, "name": "Alert System", "size": 30, "color": "#E6212B"}
}

# Add nodes
for node_id, node in nodes.items():
    fig.add_trace(go.Scatter(
        x=[node["x"]], 
        y=[node["y"]],
        mode="markers+text",
        marker=dict(size=node["size"], color=node["color"]),
        text=node["name"],
        textposition="bottom center",
        hoverinfo="text",
        name=node["name"]
    ))

# Add edges
edges = [
    ("network_devices", "monitoring"),
    ("monitoring", "data_ingest"),
    ("data_ingest", "storage"),
    ("storage", "preprocessing"),
    ("preprocessing", "feature_eng"),
    ("feature_eng", "model_training"),
    ("model_training", "model_eval"),
    ("model_eval", "prediction"),
    ("storage", "prediction"),
    ("prediction", "api"),
    ("api", "visualization"),
    ("api", "alerts"),
    ("alerts", "network_devices")  # Feedback loop
]

# Draw edges
for source, target in edges:
    fig.add_trace(go.Scatter(
        x=[nodes[source]["x"], nodes[target]["x"]],
        y=[nodes[source]["y"], nodes[target]["y"]],
        mode="lines",
        line=dict(width=2, color="#555555"),
        hoverinfo="none",
        showlegend=False
    ))

# Update layout
fig.update_layout(
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=500,
    title="Network Alarm Prediction System Architecture"
)

st.plotly_chart(fig, use_container_width=True)

# Architecture components
st.markdown("### Architecture Components")

# Create tabs for different architecture components
tab1, tab2, tab3, tab4 = st.tabs(["Data Pipeline", "ML Components", "API & Integration", "Deployment"])

with tab1:
    st.markdown("#### Data Pipeline")
    
    st.markdown("""
    The data pipeline is responsible for collecting, processing, and storing network alarm data for analysis and model training.
    
    **Key Components**:
    
    1. **Data Collection**
        - Integration with network monitoring systems via APIs or message queues
        - Polling of SNMP traps and syslog messages
        - Collection of environmental data (temperature, humidity, etc.)
        
    2. **Data Ingestion**
        - Real-time streaming data ingestion
        - Batch processing for historical data
        - Data validation and initial filtering
        
    3. **Data Storage**
        - Time-series database for efficient storage and retrieval
        - Data partitioning by time and site for optimized queries
        - Automated data retention policies
        
    4. **Data Preprocessing**
        - Cleaning and normalization of raw alarm data
        - Handling of missing values and outliers
        - Time-based aggregation and resampling
    """)
    
    # Add flowchart
    st.markdown("##### Data Flow Diagram")
    
    # Create data flow diagram with Plotly
    fig = go.Figure()
    
    # Define steps
    steps = [
        "Network<br>Devices",
        "Monitoring<br>Systems",
        "Data<br>Ingestion",
        "Validation &<br>Filtering",
        "Time-Series<br>Database",
        "Preprocessing",
        "Feature<br>Engineering"
    ]
    
    # Add steps as a horizontal flow
    for i, step in enumerate(steps):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode="markers+text",
            marker=dict(size=30, color="#0A2342"),
            text=step,
            textposition="bottom center",
            name=step
        ))
        
        # Add arrows between steps
        if i > 0:
            fig.add_annotation(
                x=i-0.5,
                y=0,
                ax=i-1,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#555555"
            )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        title="Data Pipeline Flow"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data schema
    st.markdown("##### Data Schema")
    
    # Create sample schema table
    schema = [
        {"Field": "timestamp", "Type": "datetime", "Description": "Timestamp when the alarm occurred"},
        {"Field": "alarm_id", "Type": "string", "Description": "Unique identifier for the alarm"},
        {"Field": "alarm_type", "Type": "string", "Description": "Type of alarm (e.g., Link Down, High CPU)"},
        {"Field": "site_id", "Type": "string", "Description": "Identifier for the network site"},
        {"Field": "device_id", "Type": "string", "Description": "Identifier for the specific device"},
        {"Field": "severity", "Type": "string", "Description": "Alarm severity (Critical, Major, Minor, etc.)"},
        {"Field": "temperature", "Type": "float", "Description": "Environmental temperature at the time of alarm"},
        {"Field": "humidity", "Type": "float", "Description": "Environmental humidity at the time of alarm"},
        {"Field": "duration_minutes", "Type": "integer", "Description": "Duration of the alarm condition in minutes"},
        {"Field": "resolved", "Type": "boolean", "Description": "Whether the alarm has been resolved"}
    ]
    
    # Display schema
    st.table(pd.DataFrame(schema))

with tab2:
    st.markdown("#### ML Components")
    
    st.markdown("""
    The machine learning components handle the training, evaluation, and deployment of predictive models.
    
    **Key Components**:
    
    1. **Feature Engineering**
        - Time-based feature extraction (hour, day, month, etc.)
        - Lag features for time series analysis
        - Site and device-specific features
        - Aggregation features (rolling windows, statistics)
        
    2. **Model Training**
        - Support for multiple model types (Random Forest, XGBoost, etc.)
        - Hyperparameter optimization
        - Cross-validation for robust evaluation
        - Model versioning and tracking
        
    3. **Model Evaluation**
        - Performance metrics calculation
        - Confusion matrix analysis
        - Feature importance analysis
        - A/B testing of models
        
    4. **Prediction Engine**
        - Real-time prediction service
        - Batch prediction for trend analysis
        - Confidence scoring for predictions
        - Explainability for prediction results
    """)
    
    # Model comparison
    st.markdown("##### Model Comparison")
    
    # Create sample model comparison data
    models = [
        {"Model": "Random Forest", "Accuracy": 0.85, "F1 Score": 0.83, "Training Time": "Medium", "Inference Time": "Fast"},
        {"Model": "XGBoost", "Accuracy": 0.87, "F1 Score": 0.86, "Training Time": "Medium", "Inference Time": "Fast"},
        {"Model": "Gradient Boosting", "Accuracy": 0.84, "F1 Score": 0.82, "Training Time": "Slow", "Inference Time": "Medium"},
        {"Model": "Logistic Regression", "Accuracy": 0.78, "F1 Score": 0.76, "Training Time": "Fast", "Inference Time": "Very Fast"},
        {"Model": "LSTM", "Accuracy": 0.89, "F1 Score": 0.88, "Training Time": "Very Slow", "Inference Time": "Slow"},
    ]
    
    # Display model comparison
    st.table(pd.DataFrame(models))
    
    # ML pipeline
    st.markdown("##### ML Pipeline")
    
    # Create ML pipeline diagram with Plotly
    fig = go.Figure()
    
    # Define pipeline stages
    stages = [
        {"name": "Data Splitting", "x": 0, "y": 0},
        {"name": "Feature Engineering", "x": 1, "y": 0},
        {"name": "Model Training", "x": 2, "y": 0},
        {"name": "Model Evaluation", "x": 3, "y": 0},
        {"name": "Model Deployment", "x": 4, "y": 0}
    ]
    
    # Add stages
    for stage in stages:
        fig.add_trace(go.Scatter(
            x=[stage["x"]],
            y=[stage["y"]],
            mode="markers+text",
            marker=dict(size=30, color="#F3A712"),
            text=stage["name"],
            textposition="bottom center",
            name=stage["name"]
        ))
        
        # Add arrows between stages
        if stage != stages[0]:
            prev_stage = stages[stages.index(stage) - 1]
            fig.add_annotation(
                x=(stage["x"] + prev_stage["x"]) / 2,
                y=stage["y"],
                ax=prev_stage["x"],
                ay=prev_stage["y"],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#555555"
            )
    
    # Add feedback loop
    fig.add_annotation(
        x=2,
        y=-0.5,
        ax=3.5,
        ay=-0.5,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="Hyperparameter Tuning",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#555555"
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        title="ML Pipeline"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("##### Sample Feature Importance")
    
    # Create sample feature importance data
    features = [
        {"Feature": "Previous Alarm Type", "Importance": 0.32},
        {"Feature": "Time of Day", "Importance": 0.18},
        {"Feature": "Temperature", "Importance": 0.15},
        {"Feature": "Day of Week", "Importance": 0.12},
        {"Feature": "Site ID", "Importance": 0.10},
        {"Feature": "Previous Severity", "Importance": 0.08},
        {"Feature": "Alarm Duration", "Importance": 0.05}
    ]
    
    # Create feature importance chart
    fig = px.bar(
        pd.DataFrame(features),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### API & Integration")
    
    st.markdown("""
    The API and integration components enable seamless communication between the prediction system and external systems.
    
    **Key Components**:
    
    1. **API Service**
        - RESTful API for real-time predictions
        - Batch prediction endpoints
        - Authentication and rate limiting
        - Swagger/OpenAPI documentation
        
    2. **Monitoring Integration**
        - Bidirectional integration with network monitoring systems
        - SNMP trap receivers
        - Syslog integration
        - Webhook support for alerting systems
        
    3. **Visualization Service**
        - Interactive dashboards
        - Real-time visualization updates
        - Custom reporting capabilities
        - Data export functionality
        
    4. **Alert System**
        - Configurable alert thresholds
        - Multiple notification channels (email, SMS, etc.)
        - Alert prioritization based on severity
        - Alert deduplication and correlation
    """)
    
    # API endpoints
    st.markdown("##### API Endpoints")
    
    # Create sample API endpoint documentation
    endpoints = [
        {
            "Endpoint": "/api/v1/predictions/next",
            "Method": "POST",
            "Description": "Predict the next alarm for a given site",
            "Request": "{ site_id: string, timestamp: datetime }",
            "Response": "{ prediction: { alarm_type: string, probability: float, estimated_time: datetime } }"
        },
        {
            "Endpoint": "/api/v1/predictions/batch",
            "Method": "POST",
            "Description": "Generate batch predictions for multiple sites",
            "Request": "{ sites: [string], time_range: { start: datetime, end: datetime } }",
            "Response": "{ predictions: [{ site_id: string, predictions: [{ alarm_type, probability, time }] }] }"
        },
        {
            "Endpoint": "/api/v1/alarms",
            "Method": "GET",
            "Description": "Retrieve historical alarm data with filtering",
            "Request": "?site_id=string&start_time=datetime&end_time=datetime&alarm_type=string",
            "Response": "{ alarms: [{ alarm_id, timestamp, site_id, alarm_type, severity, ... }] }"
        },
        {
            "Endpoint": "/api/v1/sites/{site_id}/health",
            "Method": "GET",
            "Description": "Get the current health score and status for a site",
            "Request": "Path parameter: site_id",
            "Response": "{ site_id: string, health_score: float, status: string, issues: [string] }"
        },
        {
            "Endpoint": "/api/v1/models/status",
            "Method": "GET",
            "Description": "Get the status of trained models",
            "Request": "None",
            "Response": "{ models: [{ model_id, type, accuracy, last_trained, status }] }"
        }
    ]
    
    # Display API endpoints
    st.table(pd.DataFrame(endpoints))
    
    # Integration diagram
    st.markdown("##### Integration Architecture")
    
    # Create integration diagram with Plotly
    fig = go.Figure()
    
    # Define system nodes
    systems = [
        {"name": "Network Monitoring", "x": 0, "y": 0, "color": "#0A2342"},
        {"name": "Alarm Prediction", "x": 2, "y": 0, "color": "#F3A712"},
        {"name": "Ticketing System", "x": 4, "y": 0, "color": "#2CA58D"},
        {"name": "Notification System", "x": 2, "y": -2, "color": "#E6212B"},
        {"name": "Dashboard", "x": 2, "y": 2, "color": "#2CA58D"}
    ]
    
    # Add systems
    for system in systems:
        fig.add_trace(go.Scatter(
            x=[system["x"]],
            y=[system["y"]],
            mode="markers+text",
            marker=dict(size=40, color=system["color"]),
            text=system["name"],
            textposition="middle center",
            name=system["name"]
        ))
    
    # Add connections
    connections = [
        (0, 1, "Alarm Data"),
        (1, 0, "Predictions"),
        (1, 2, "Auto Tickets"),
        (1, 3, "Alerts"),
        (1, 4, "Visualizations"),
        (3, 0, "Notifications")
    ]
    
    for source, target, label in connections:
        source_system = systems[source]
        target_system = systems[target]
        
        # Calculate midpoint for label
        mid_x = (source_system["x"] + target_system["x"]) / 2
        mid_y = (source_system["y"] + target_system["y"]) / 2
        
        # Add line
        fig.add_trace(go.Scatter(
            x=[source_system["x"], target_system["x"]],
            y=[source_system["y"], target_system["y"]],
            mode="lines",
            line=dict(width=2, color="#555555"),
            showlegend=False
        ))
        
        # Add label
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=label,
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        title="System Integration Architecture"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("#### Deployment Architecture")
    
    st.markdown("""
    The deployment architecture describes how the system is deployed and operated in production.
    
    **Key Components**:
    
    1. **Infrastructure**
        - Containerized microservices architecture
        - Kubernetes for orchestration
        - Cloud-native deployment (AWS, Azure, GCP)
        - High availability configuration
        
    2. **Scaling**
        - Horizontal scaling for API services
        - Vertical scaling for ML training
        - Auto-scaling based on traffic patterns
        - Resource optimization
        
    3. **Security**
        - API authentication and authorization
        - Data encryption (in transit and at rest)
        - Network segmentation
        - Compliance with security standards
        
    4. **Monitoring & Operations**
        - System health monitoring
        - Performance metrics
        - Centralized logging
        - Automated backup and recovery
    """)
    
    # Deployment diagram
    st.markdown("##### Deployment Diagram")
    
    # Create deployment diagram with Plotly
    fig = go.Figure()
    
    # Define components
    components = [
        {"name": "Load Balancer", "x": 0, "y": 0, "width": 1, "height": 1, "color": "#0A2342"},
        {"name": "API Service", "x": 2, "y": -1, "width": 1, "height": 1, "color": "#2CA58D"},
        {"name": "API Service", "x": 2, "y": 0, "width": 1, "height": 1, "color": "#2CA58D"},
        {"name": "API Service", "x": 2, "y": 1, "width": 1, "height": 1, "color": "#2CA58D"},
        {"name": "Prediction Engine", "x": 4, "y": -1, "width": 1, "height": 1, "color": "#F3A712"},
        {"name": "Prediction Engine", "x": 4, "y": 1, "width": 1, "height": 1, "color": "#F3A712"},
        {"name": "Model Training", "x": 6, "y": 0, "width": 1, "height": 1, "color": "#F3A712"},
        {"name": "Time-Series DB", "x": 8, "y": -1, "width": 1, "height": 1, "color": "#0A2342"},
        {"name": "Model Registry", "x": 8, "y": 1, "width": 1, "height": 1, "color": "#0A2342"}
    ]
    
    # Add rectangles for components
    for component in components:
        fig.add_shape(
            type="rect",
            x0=component["x"] - component["width"]/2,
            y0=component["y"] - component["height"]/2,
            x1=component["x"] + component["width"]/2,
            y1=component["y"] + component["height"]/2,
            line=dict(color="#000000", width=1),
            fillcolor=component["color"],
            opacity=0.7
        )
        
        # Add component name
        fig.add_annotation(
            x=component["x"],
            y=component["y"],
            text=component["name"],
            showarrow=False,
            font=dict(color="white", size=10)
        )
    
    # Add connections
    connections = [
        (0, 2),  # Load Balancer to middle API Service
        (0, 1),  # Load Balancer to top API Service
        (0, 3),  # Load Balancer to bottom API Service
        (1, 4),  # Top API Service to top Prediction Engine
        (2, 4),  # Middle API Service to top Prediction Engine
        (3, 5),  # Bottom API Service to bottom Prediction Engine
        (2, 5),  # Middle API Service to bottom Prediction Engine
        (4, 6),  # Top Prediction Engine to Model Training
        (5, 6),  # Bottom Prediction Engine to Model Training
        (6, 7),  # Model Training to Time-Series DB
        (6, 8),  # Model Training to Model Registry
        (4, 8),  # Top Prediction Engine to Model Registry
        (5, 8),  # Bottom Prediction Engine to Model Registry
        (4, 7),  # Top Prediction Engine to Time-Series DB
        (5, 7)   # Bottom Prediction Engine to Time-Series DB
    ]
    
    # Add lines for connections
    for source, target in connections:
        source_component = components[source]
        target_component = components[target]
        
        fig.add_trace(go.Scatter(
            x=[source_component["x"], target_component["x"]],
            y=[source_component["y"], target_component["y"]],
            mode="lines",
            line=dict(width=1, color="#555555"),
            showlegend=False
        ))
    
    # Add kubernetes cluster boundary
    fig.add_shape(
        type="rect",
        x0=-1,
        y0=-2,
        x1=9,
        y1=2,
        line=dict(color="#0A2342", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Add cluster label
    fig.add_annotation(
        x=-0.8,
        y=1.8,
        text="Kubernetes Cluster",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="#0A2342")
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        title="Deployment Architecture",
        plot_bgcolor="white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technology stack
    st.markdown("##### Technology Stack")
    
    # Create columns for different technology categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Infrastructure & Deployment**")
        st.markdown("- Docker containers")
        st.markdown("- Kubernetes orchestration")
        st.markdown("- Helm charts for deployment")
        st.markdown("- Terraform for infrastructure")
        st.markdown("- CI/CD with GitHub Actions")
    
    with col2:
        st.markdown("**Backend & Data Processing**")
        st.markdown("- Python for core logic")
        st.markdown("- FastAPI for REST endpoints")
        st.markdown("- Pandas for data manipulation")
        st.markdown("- TimescaleDB for time-series data")
        st.markdown("- Redis for caching")
    
    with col3:
        st.markdown("**ML & Analytics**")
        st.markdown("- Scikit-learn for modeling")
        st.markdown("- XGBoost for gradient boosting")
        st.markdown("- MLflow for model tracking")
        st.markdown("- Streamlit for visualization")
        st.markdown("- Plotly for interactive charts")

# Technical specifications
st.markdown("## 📝 Technical Specifications")

# Create tabs for different specifications
tab1, tab2, tab3 = st.tabs(["Performance Requirements", "Scalability", "Security & Compliance"])

with tab1:
    st.markdown("### Performance Requirements")
    
    st.markdown("""
    The system is designed to meet the following performance requirements:
    
    **Response Time**:
    - API response time < 200ms for prediction requests
    - Dashboard loading time < 2 seconds
    - Alert generation time < 500ms after prediction
    
    **Throughput**:
    - Support for 1000+ prediction requests per minute
    - Ability to process 10,000+ alarms per minute
    - Support for 100+ concurrent dashboard users
    
    **Accuracy**:
    - Prediction accuracy > 85% for next alarm type
    - False positive rate < 5%
    - Precision > 80% for critical alarms
    
    **Availability**:
    - 99.9% uptime for prediction API
    - 99.5% uptime for dashboard services
    - Automated failover in case of component failure
    """)
    
    # Performance metrics
    st.markdown("#### Performance Metrics")
    
    # Create sample performance metrics
    metrics = [
        {"Metric": "API Response Time", "Target": "< 200ms", "Current": "175ms", "Status": "✅ Meeting Target"},
        {"Metric": "Prediction Accuracy", "Target": "> 85%", "Current": "87%", "Status": "✅ Meeting Target"},
        {"Metric": "False Positive Rate", "Target": "< 5%", "Current": "4.2%", "Status": "✅ Meeting Target"},
        {"Metric": "Alert Generation Time", "Target": "< 500ms", "Current": "350ms", "Status": "✅ Meeting Target"},
        {"Metric": "Dashboard Loading Time", "Target": "< 2s", "Current": "1.8s", "Status": "✅ Meeting Target"}
    ]
    
    # Display metrics
    st.table(pd.DataFrame(metrics))

with tab2:
    st.markdown("### Scalability")
    
    st.markdown("""
    The system is designed to scale horizontally and vertically to handle growing data volumes and user load.
    
    **Horizontal Scaling**:
    - API services can scale to multiple instances
    - Prediction engine supports distributed processing
    - Database supports clustering and sharding
    
    **Vertical Scaling**:
    - Model training can utilize high-performance compute resources
    - Database instances can be scaled up for improved performance
    - Batch processing jobs can utilize additional resources on demand
    
    **Data Volume Handling**:
    - Support for billions of alarm records
    - Time-based partitioning for efficient data access
    - Automated data archiving for historical records
    
    **Load Balancing**:
    - Automatic request distribution across API instances
    - Health-based routing to avoid overloaded instances
    - Geographic distribution for global deployments
    """)
    
    # Scalability metrics
    st.markdown("#### Scalability Metrics")
    
    # Create sample scalability metrics
    metrics = [
        {"Scale Factor": "1x", "API Instances": "3", "Prediction Instances": "2", "Max Requests/min": "1,000", "Max Alarms/min": "10,000"},
        {"Scale Factor": "2x", "API Instances": "6", "Prediction Instances": "4", "Max Requests/min": "2,000", "Max Alarms/min": "20,000"},
        {"Scale Factor": "5x", "API Instances": "15", "Prediction Instances": "10", "Max Requests/min": "5,000", "Max Alarms/min": "50,000"},
        {"Scale Factor": "10x", "API Instances": "30", "Prediction Instances": "20", "Max Requests/min": "10,000", "Max Alarms/min": "100,000"}
    ]
    
    # Display metrics
    st.table(pd.DataFrame(metrics))

with tab3:
    st.markdown("### Security & Compliance")
    
    st.markdown("""
    The system implements comprehensive security measures and complies with relevant standards.
    
    **Authentication & Authorization**:
    - OAuth 2.0 / OpenID Connect for user authentication
    - Role-based access control (RBAC)
    - API key management for service-to-service communication
    - JWT token-based authentication for API requests
    
    **Data Security**:
    - Encryption of data in transit (TLS 1.3)
    - Encryption of sensitive data at rest
    - Data anonymization for non-production environments
    - Secure key management
    
    **Network Security**:
    - Network segmentation with security groups
    - Web Application Firewall (WAF) protection
    - DDoS protection
    - Rate limiting for API endpoints
    
    **Compliance**:
    - GDPR compliance for personal data
    - SOC 2 compliance for service operation
    - Regular security audits and penetration testing
    - Comprehensive logging for audit trails
    """)
    
    # Security checklist
    st.markdown("#### Security Checklist")
    
    # Create sample security checklist
    checklist = [
        {"Requirement": "TLS 1.3 for all communications", "Status": "✅ Implemented"},
        {"Requirement": "API authentication", "Status": "✅ Implemented"},
        {"Requirement": "Role-based access control", "Status": "✅ Implemented"},
        {"Requirement": "Data encryption at rest", "Status": "✅ Implemented"},
        {"Requirement": "Vulnerability scanning", "Status": "✅ Implemented"},
        {"Requirement": "Security logging", "Status": "✅ Implemented"},
        {"Requirement": "Regular penetration testing", "Status": "✅ Implemented"},
        {"Requirement": "Data anonymization", "Status": "✅ Implemented"}
    ]
    
    # Display checklist
    st.table(pd.DataFrame(checklist))

# Implementation roadmap
st.markdown("## 🗓️ Implementation Roadmap")

# Create roadmap chart
st.markdown("### Project Timeline")

# Create sample milestone data
milestones = [
    {"Phase": "Phase 1", "Milestone": "Initial Data Pipeline", "Start": "2025-01-01", "End": "2025-02-15", "Status": "Completed"},
    {"Phase": "Phase 1", "Milestone": "Basic ML Models", "Start": "2025-02-01", "End": "2025-03-15", "Status": "Completed"},
    {"Phase": "Phase 1", "Milestone": "MVP Dashboard", "Start": "2025-03-01", "End": "2025-04-15", "Status": "In Progress"},
    {"Phase": "Phase 2", "Milestone": "Advanced Feature Engineering", "Start": "2025-04-01", "End": "2025-05-15", "Status": "Planned"},
    {"Phase": "Phase 2", "Milestone": "Improved Model Accuracy", "Start": "2025-05-01", "End": "2025-06-15", "Status": "Planned"},
    {"Phase": "Phase 2", "Milestone": "API Integration", "Start": "2025-05-15", "End": "2025-07-15", "Status": "Planned"},
    {"Phase": "Phase 3", "Milestone": "Real-time Predictions", "Start": "2025-07-01", "End": "2025-08-15", "Status": "Planned"},
    {"Phase": "Phase 3", "Milestone": "Alert System", "Start": "2025-08-01", "End": "2025-09-15", "Status": "Planned"},
    {"Phase": "Phase 3", "Milestone": "Production Deployment", "Start": "2025-09-01", "End": "2025-10-15", "Status": "Planned"}
]

# Convert dates to datetime
for milestone in milestones:
    milestone["Start"] = pd.to_datetime(milestone["Start"])
    milestone["End"] = pd.to_datetime(milestone["End"])
    milestone["Duration"] = (milestone["End"] - milestone["Start"]).days

# Create DataFrame
milestones_df = pd.DataFrame(milestones)

# Define colors for status
color_map = {
    "Completed": "#4CAF50",
    "In Progress": "#2196F3",
    "Planned": "#9E9E9E"
}

# Create Gantt chart
fig = px.timeline(
    milestones_df,
    x_start="Start",
    x_end="End",
    y="Milestone",
    color="Status",
    color_discrete_map=color_map,
    hover_data=["Phase", "Duration"],
    title="Project Implementation Timeline"
)

# Update layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Milestone",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Future enhancements
st.markdown("### Future Enhancements")

# Create columns for different enhancement categories
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Phase 4 (2025 Q4)")
    st.markdown("- Integration with additional monitoring systems")
    st.markdown("- Advanced anomaly detection algorithms")
    st.markdown("- Auto-remediation for common issues")
    st.markdown("- Mobile application for on-the-go alerts")
    st.markdown("- Enhanced root cause analysis")

with col2:
    st.markdown("#### Phase 5 (2026 Q1-Q2)")
    st.markdown("- AI-powered network optimization")
    st.markdown("- Predictive capacity planning")
    st.markdown("- Integration with service management systems")
    st.markdown("- Multi-tenant support")
    st.markdown("- Global deployment with geo-redundancy")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Insights](/pages/6_insights.py)")

with col2:
    st.markdown("[Continue to Demo Simulation →](/pages/8_demo.py)")