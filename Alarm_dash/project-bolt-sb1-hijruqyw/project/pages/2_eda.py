import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import visualization utilities
from utils.visualization import (
    plot_alarm_distribution, 
    plot_alarm_severity, 
    plot_alarm_trends,
    plot_site_heatmap,
    plot_correlation_matrix,
    plot_alarm_calendar
)

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.stop()

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">📈</div>
        <div class="title-container">
            <h1 class="main-title">Exploratory Data Analysis</h1>
            <p class="subtitle">Discover patterns and insights in your network alarm data</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Make a copy of the data for analysis
df = st.session_state.data.copy()

# Convert timestamp to datetime if needed
if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar for filtering
st.sidebar.markdown("## 🔍 Data Filters")

# Time range filter
st.sidebar.markdown("### Time Range")
min_date = df['timestamp'].min().date() if 'timestamp' in df.columns else datetime.now().date()
max_date = df['timestamp'].max().date() if 'timestamp' in df.columns else datetime.now().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Site filter
st.sidebar.markdown("### Sites")
all_sites = df['site_id'].unique()
selected_sites = st.sidebar.multiselect("Select Sites", all_sites, default=all_sites[:5] if len(all_sites) > 5 else all_sites)

# Alarm type filter
st.sidebar.markdown("### Alarm Types")
all_alarm_types = df['alarm_type'].unique()
selected_alarm_types = st.sidebar.multiselect("Select Alarm Types", all_alarm_types, default=all_alarm_types)

# Severity filter
st.sidebar.markdown("### Severity")
all_severities = df['severity'].unique()
selected_severities = st.sidebar.multiselect("Select Severities", all_severities, default=all_severities)

# Apply filters
filtered_df = df.copy()

# Filter by date
filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & 
                          (filtered_df['timestamp'].dt.date <= end_date)]

# Filter by site
if selected_sites:
    filtered_df = filtered_df[filtered_df['site_id'].isin(selected_sites)]

# Filter by alarm type
if selected_alarm_types:
    filtered_df = filtered_df[filtered_df['alarm_type'].isin(selected_alarm_types)]

# Filter by severity
if selected_severities:
    filtered_df = filtered_df[filtered_df['severity'].isin(selected_severities)]

# Display filter summary
st.markdown("## 📋 Data Overview")
st.markdown(f"**Filtered Data:** {filtered_df.shape[0]} rows from {start_date} to {end_date}")

# Create metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_alarms = filtered_df.shape[0]
    st.metric("Total Alarms", total_alarms)

with col2:
    unique_sites = filtered_df['site_id'].nunique()
    st.metric("Unique Sites", unique_sites)

with col3:
    alarm_types_count = filtered_df['alarm_type'].nunique()
    st.metric("Alarm Types", alarm_types_count)

with col4:
    if 'severity' in filtered_df.columns:
        critical_count = filtered_df[filtered_df['severity'] == 'Critical'].shape[0]
        critical_pct = (critical_count / total_alarms) * 100 if total_alarms > 0 else 0
        st.metric("Critical Alarms", f"{critical_count} ({critical_pct:.1f}%)")

# Create visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["Alarm Distribution", "Time Trends", "Site Analysis", "Correlations"])

with tab1:
    st.markdown("## 📊 Alarm Distribution Analysis")
    
    # Alarm type distribution
    st.markdown("### Alarm Type Distribution")
    fig = plot_alarm_distribution(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Severity distribution
    st.markdown("### Severity Distribution")
    fig = plot_alarm_severity(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show raw counts
    with st.expander("Show detailed counts"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Alarm Type Counts")
            alarm_counts = filtered_df['alarm_type'].value_counts().reset_index()
            alarm_counts.columns = ['Alarm Type', 'Count']
            st.dataframe(alarm_counts, use_container_width=True)
        
        with col2:
            st.markdown("#### Severity Counts")
            severity_counts = filtered_df['severity'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            st.dataframe(severity_counts, use_container_width=True)

with tab2:
    st.markdown("## ⏱️ Time Trend Analysis")
    
    # Alarm trends over time
    st.markdown("### Alarm Trends Over Time")
    fig = plot_alarm_trends(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calendar heatmap
    st.markdown("### Alarm Calendar Heatmap")
    fig = plot_alarm_calendar(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time of day analysis
    st.markdown("### Alarms by Hour of Day")
    
    # Extract hour of day
    if 'hour' not in filtered_df.columns:
        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    
    # Create hourly counts
    hourly_counts = filtered_df.groupby('hour').size().reset_index(name='count')
    
    # Create bar chart
    fig = px.bar(
        hourly_counts, 
        x='hour', 
        y='count',
        title='Alarms by Hour of Day',
        labels={'hour': 'Hour (24-hour format)', 'count': 'Number of Alarms'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 🌐 Site Analysis")
    
    # Alarms by site
    st.markdown("### Alarms by Site")
    site_counts = filtered_df.groupby('site_id').size().reset_index(name='count')
    site_counts = site_counts.sort_values('count', ascending=False)
    
    fig = px.bar(
        site_counts, 
        x='site_id', 
        y='count',
        title='Number of Alarms by Site',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alarm heatmap by site and type
    st.markdown("### Alarm Heatmap by Site and Type")
    fig = plot_site_heatmap(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Site severity analysis
    st.markdown("### Severity by Site")
    
    # Calculate severity percentages by site
    severity_by_site = filtered_df.groupby(['site_id', 'severity']).size().reset_index(name='count')
    site_totals = severity_by_site.groupby('site_id')['count'].sum().reset_index(name='total')
    severity_by_site = severity_by_site.merge(site_totals, on='site_id')
    severity_by_site['percentage'] = (severity_by_site['count'] / severity_by_site['total']) * 100
    
    # Create stacked bar chart
    fig = px.bar(
        severity_by_site,
        x='site_id',
        y='percentage',
        color='severity',
        title='Severity Distribution by Site',
        labels={'percentage': 'Percentage of Alarms', 'site_id': 'Site ID'},
        color_discrete_map={
            'Critical': '#FF5252',
            'Major': '#FF9800',
            'Minor': '#FFC107',
            'Warning': '#8BC34A',
            'Info': '#2196F3'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## 🔗 Correlation Analysis")
    
    # Identify numeric columns for correlation
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Correlation matrix
        st.markdown("### Feature Correlation Matrix")
        fig = plot_correlation_matrix(filtered_df, numeric_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot for selected features
        st.markdown("### Feature Relationship Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", numeric_cols)
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", [col for col in numeric_cols if col != x_feature], index=min(1, len(numeric_cols) - 1))
        
        color_by = st.selectbox("Color by", ['alarm_type', 'severity', 'site_id'])
        
        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=color_by,
            title=f'Relationship between {x_feature} and {y_feature}',
            hover_data=['timestamp', 'alarm_type', 'site_id', 'severity']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric features available for correlation analysis.")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Data Upload](/pages/1_data_upload.py)")

with col2:
    st.markdown("[Continue to ML Modeling →](/pages/3_modeling.py)")