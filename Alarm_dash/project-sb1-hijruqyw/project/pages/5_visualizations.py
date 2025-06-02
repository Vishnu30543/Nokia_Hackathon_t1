import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.markdown("[← Go to Data Upload](/pages/1_data_upload.py)")
    st.stop()

# Import visualization utilities
from utils.visualization import (
    plot_alarm_distribution,
    plot_alarm_severity,
    plot_alarm_trends,
    plot_site_heatmap,
    plot_hotspot_map,
    plot_alarm_calendar,
    plot_correlation_matrix
)

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">📈</div>
        <div class="title-container">
            <h1 class="main-title">Advanced Visualizations</h1>
            <p class="subtitle">Interactive visual analysis of network alarm patterns</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Make a copy of the data
df = st.session_state.data.copy()
print(df)
# Convert timestamp to datetime if needed
if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar for visualization options
st.sidebar.markdown("## 🔍 Visualization Options")

# Visualization type selection
viz_type = st.sidebar.selectbox(
    "Select Visualization Type",
    [
        "Alarm Hotspot Map",
        "Time-based Analysis",
        "Site Comparison",
        "Severity Analysis",
        "Correlation Insights",
        "Custom Dashboard"
    ]
)

# Common filters
st.sidebar.markdown("### Data Filters")

# Time range filter
st.sidebar.markdown("#### Time Range")
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Apply time filter
filtered_df = df[(df['timestamp'].dt.date >= start_date) &
                (df['timestamp'].dt.date <= end_date)]

# Site filter
all_sites = df['site_id'].unique()
selected_sites = st.sidebar.multiselect(
    "Select Sites",
    all_sites,
    default=all_sites[:5] if len(all_sites) > 5 else all_sites
)

# Apply site filter if selected
if selected_sites:
    filtered_df = filtered_df[filtered_df['site_id'].isin(selected_sites)]

# Alarm type filter
all_alarm_types = df['alarm_type'].unique()
selected_alarm_types = st.sidebar.multiselect(
    "Select Alarm Types",
    all_alarm_types,
    default=[]
)

# Apply alarm type filter if selected
if selected_alarm_types:
    filtered_df = filtered_df[filtered_df['alarm_type'].isin(selected_alarm_types)]

# Display visualizations based on selection
if viz_type == "Alarm Hotspot Map":
    st.markdown("## 🌐 Network Alarm Hotspot Map")

    st.markdown("""
    This visualization shows the distribution of alarms across the network, highlighting hotspots
    where alarms are more frequent or severe. Larger circles represent sites with more alarms,
    while color indicates the average severity level.
    """)

    # Create hotspot map
    fig = plot_hotspot_map(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Alarm Sites")

        # Count alarms by site
        site_counts = filtered_df.groupby('site_id').size().reset_index(name='count')
        site_counts = site_counts.sort_values('count', ascending=False).head(10)

        # Create bar chart
        fig = px.bar(
            site_counts,
            x='site_id',
            y='count',
            title='Top 10 Sites by Alarm Count',
            color='count',
            color_continuous_scale='Viridis'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Critical Alarm Distribution")

        # Filter for critical alarms
        critical_df = filtered_df[filtered_df['severity'] == 'Critical']

        # Count critical alarms by site
        critical_counts = critical_df.groupby('site_id').size().reset_index(name='count')
        critical_counts = critical_counts.sort_values('count', ascending=False).head(10)

        # Create bar chart
        fig = px.bar(
            critical_counts,
            x='site_id',
            y='count',
            title='Top 10 Sites by Critical Alarm Count',
            color='count',
            color_continuous_scale='Reds'
        )

        st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Time-based Analysis":
    st.markdown("## ⏱️ Time-based Alarm Analysis")

    # Create tabs for different time analyses
    tab1, tab2, tab3 = st.tabs(["Trends Over Time", "Hour of Day Analysis", "Calendar View"])

    with tab1:
        st.markdown("### Alarm Trends Over Time")

        # Create trend plot
        fig = plot_alarm_trends(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Trend analysis
        st.markdown("#### Trend Analysis")

        # Resample by day
        daily_counts = filtered_df.copy()
        daily_counts['date'] = daily_counts['timestamp'].dt.date
        daily_counts = daily_counts.groupby('date').size().reset_index(name='count')

        # Calculate moving average
        window_size = st.slider("Moving Average Window (days)", 1, 30, 7)
        daily_counts['moving_avg'] = daily_counts['count'].rolling(window=window_size).mean()

        # Create trend plot with moving average
        fig = px.line(
            daily_counts,
            x='date',
            y=['count', 'moving_avg'],
            title=f'Alarm Trend with {window_size}-day Moving Average',
            labels={'value': 'Number of Alarms', 'date': 'Date', 'variable': 'Metric'},
            color_discrete_map={'count': 'blue', 'moving_avg': 'red'}
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Hour of Day Analysis")

        # Extract hour of day
        hour_df = filtered_df.copy()
        hour_df['hour'] = hour_df['timestamp'].dt.hour

        # Count alarms by hour
        hourly_counts = hour_df.groupby('hour').size().reset_index(name='count')

        # Create hour plot
        fig = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            title='Alarms by Hour of Day',
            color='count',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title='Hour of Day (24-hour format)',
            yaxis_title='Number of Alarms'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Hour analysis by alarm type
        st.markdown("#### Hourly Patterns by Alarm Type")

        # Count alarms by hour and type
        hourly_type_counts = hour_df.groupby(['hour', 'alarm_type']).size().reset_index(name='count')

        # Create hour by type plot
        fig = px.line(
            hourly_type_counts,
            x='hour',
            y='count',
            color='alarm_type',
            title='Alarms by Hour and Type',
            line_shape='spline'
        )

        fig.update_layout(
            xaxis_title='Hour of Day (24-hour format)',
            yaxis_title='Number of Alarms'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Calendar Heatmap")

        # Create calendar heatmap
        fig = plot_alarm_calendar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Day of week analysis
        st.markdown("#### Day of Week Analysis")

        # Extract day of week
        dow_df = filtered_df.copy()
        dow_df['day_of_week'] = dow_df['timestamp'].dt.dayofweek

        # Count alarms by day of week
        dow_counts = dow_df.groupby('day_of_week').size().reset_index(name='count')

        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts['day_name'] = dow_counts['day_of_week'].map(lambda x: day_names[x])

        # Create day of week plot
        fig = px.bar(
            dow_counts,
            x='day_name',
            y='count',
            title='Alarms by Day of Week',
            color='count',
            color_continuous_scale='Viridis'
        )

        # Set category order
        fig.update_layout(
            xaxis=dict(
                categoryorder='array',
                categoryarray=day_names
            )
        )

        st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Site Comparison":
    st.markdown("## 🔄 Site Comparison Analysis")

    # Site selection for comparison
    if len(selected_sites) < 2:
        st.warning("Please select at least two sites for comparison.")

        # Suggest some sites
        top_sites = df.groupby('site_id').size().sort_values(ascending=False).head(5).index.tolist()

        st.markdown(f"Suggested sites for comparison: {', '.join(top_sites)}")
    else:
        st.markdown(f"### Comparing {len(selected_sites)} Selected Sites")

        # Create comparison metrics
        st.markdown("#### Key Metrics Comparison")

        # Calculate metrics by site
        site_metrics = []

        for site in selected_sites:
            site_df = filtered_df[filtered_df['site_id'] == site]

            if not site_df.empty:
                # Calculate metrics
                total_alarms = site_df.shape[0]
                critical_alarms = site_df[site_df['severity'] == 'Critical'].shape[0] if 'severity' in site_df.columns else 0
                critical_pct = (critical_alarms / total_alarms) * 100 if total_alarms > 0 else 0

                # Calculate unique alarm types
                unique_alarms = site_df['alarm_type'].nunique()

                # Calculate average temperature if available
                avg_temp = site_df['temperature'].mean() if 'temperature' in site_df.columns else None

                # Add to metrics list
                site_metrics.append({
                    'site_id': site,
                    'total_alarms': total_alarms,
                    'critical_alarms': critical_alarms,
                    'critical_pct': critical_pct,
                    'unique_alarm_types': unique_alarms,
                    'avg_temperature': avg_temp
                })

        # Convert to DataFrame
        metrics_df = pd.DataFrame(site_metrics)

        # Create radar chart
        if not metrics_df.empty:
            # Normalize metrics for radar chart
            radar_df = metrics_df.copy()

            for col in ['total_alarms', 'critical_alarms', 'unique_alarm_types']:
                max_val = radar_df[col].max()
                if max_val > 0:
                    radar_df[f'{col}_norm'] = radar_df[col] / max_val
                else:
                    radar_df[f'{col}_norm'] = 0

            # Create radar chart
            fig = go.Figure()

            # Add traces for each site
            for site in radar_df['site_id']:
                site_row = radar_df[radar_df['site_id'] == site].iloc[0]

                fig.add_trace(go.Scatterpolar(
                    r=[
                        site_row['total_alarms_norm'],
                        site_row['critical_alarms_norm'],
                        site_row['critical_pct'] / 100,
                        site_row['unique_alarm_types_norm']
                    ],
                    theta=['Total Alarms', 'Critical Alarms', 'Critical %', 'Alarm Types'],
                    fill='toself',
                    name=site
                ))

            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='Site Comparison Radar Chart'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Create comparison table
        st.dataframe(metrics_df, use_container_width=True)

        # Alarm type comparison
        st.markdown("#### Alarm Type Distribution by Site")

        # Count alarms by site and type
        site_type_counts = filtered_df.groupby(['site_id', 'alarm_type']).size().reset_index(name='count')

        # Filter for selected sites
        site_type_counts = site_type_counts[site_type_counts['site_id'].isin(selected_sites)]

        # Create stacked bar chart
        fig = px.bar(
            site_type_counts,
            x='site_id',
            y='count',
            color='alarm_type',
            title='Alarm Types by Site',
            barmode='stack'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Severity comparison
        st.markdown("#### Severity Distribution by Site")

        # Count alarms by site and severity
        site_severity_counts = filtered_df.groupby(['site_id', 'severity']).size().reset_index(name='count')

        # Filter for selected sites
        site_severity_counts = site_severity_counts[site_severity_counts['site_id'].isin(selected_sites)]

        # Create stacked bar chart
        fig = px.bar(
            site_severity_counts,
            x='site_id',
            y='count',
            color='severity',
            title='Alarm Severity by Site',
            barmode='stack',
            color_discrete_map={
                'Critical': '#FF5252',
                'Major': '#FF9800',
                'Minor': '#FFC107',
                'Warning': '#8BC34A',
                'Info': '#2196F3'
            }
        )

        st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Severity Analysis":
    st.markdown("## 🚨 Severity Analysis")

    # Severity distribution
    st.markdown("### Alarm Severity Distribution")
    fig = plot_alarm_severity(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

    # Severity trends over time
    st.markdown("### Severity Trends Over Time")

    # Resample by day
    severity_df = filtered_df.copy()
    severity_df['date'] = severity_df['timestamp'].dt.date

    # Count alarms by date and severity
    daily_severity = severity_df.groupby(['date', 'severity']).size().reset_index(name='count')

    # Create line plot
    fig = px.line(
        daily_severity,
        x='date',
        y='count',
        color='severity',
        title='Severity Trends Over Time',
        line_shape='spline',
        color_discrete_map={
            'Critical': '#FF5252',
            'Major': '#FF9800',
            'Minor': '#FFC107',
            'Warning': '#8BC34A',
            'Info': '#2196F3'
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # Severity by alarm type
    st.markdown("### Severity by Alarm Type")

    # Count alarms by type and severity
    type_severity = filtered_df.groupby(['alarm_type', 'severity']).size().reset_index(name='count')

    # Create stacked bar chart
    fig = px.bar(
        type_severity,
        x='alarm_type',
        y='count',
        color='severity',
        title='Severity Distribution by Alarm Type',
        barmode='stack',
        color_discrete_map={
            'Critical': '#FF5252',
            'Major': '#FF9800',
            'Minor': '#FFC107',
            'Warning': '#8BC34A',
            'Info': '#2196F3'
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # Severity distribution by hour
    st.markdown("### Severity Distribution by Hour of Day")

    # Extract hour
    hour_severity = filtered_df.copy()
    hour_severity['hour'] = hour_severity['timestamp'].dt.hour

    # Count alarms by hour and severity
    hourly_severity = hour_severity.groupby(['hour', 'severity']).size().reset_index(name='count')

    # Create stacked bar chart
    fig = px.bar(
        hourly_severity,
        x='hour',
        y='count',
        color='severity',
        title='Severity Distribution by Hour of Day',
        barmode='stack',
        color_discrete_map={
            'Critical': '#FF5252',
            'Major': '#FF9800',
            'Minor': '#FFC107',
            'Warning': '#8BC34A',
            'Info': '#2196F3'
        }
    )

    fig.update_layout(
        xaxis_title='Hour of Day (24-hour format)',
        yaxis_title='Number of Alarms'
    )

    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Correlation Insights":
    st.markdown("## 🔗 Correlation Insights")

    # Check for numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Not enough numeric features for correlation analysis.")
    else:
        # Correlation matrix
        st.markdown("### Feature Correlation Matrix")

        # Create correlation matrix
        fig = plot_correlation_matrix(filtered_df, numeric_cols)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot matrix
        st.markdown("### Scatter Plot Matrix")

        # Select columns for scatter matrix
        default_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        selected_cols = st.multiselect(
            "Select Features for Scatter Matrix",
            numeric_cols,
            default=default_cols
        )

        if len(selected_cols) < 2:
            st.warning("Please select at least two features for the scatter matrix.")
        else:
            # Create scatter matrix
            fig = px.scatter_matrix(
                filtered_df,
                dimensions=selected_cols,
                color='alarm_type' if 'alarm_type' in filtered_df.columns else None,
                title='Feature Scatter Matrix'
            )

            fig.update_layout(height=800)

            st.plotly_chart(fig, use_container_width=True)

        # Feature relationship explorer
        st.markdown("### Feature Relationship Explorer")

        # Select x and y features
        col1, col2, col3 = st.columns(3)

        with col1:
            x_feature = st.selectbox("X-axis Feature", numeric_cols, index=0)

        with col2:
            y_feature = st.selectbox(
                "Y-axis Feature",
                [col for col in numeric_cols if col != x_feature],
                index=min(1, len(numeric_cols) - 1)
            )

        with col3:
            color_by = st.selectbox(
                "Color by",
                ['alarm_type', 'severity', 'site_id'],
                index=0
            )

        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color=color_by,
            title=f'Relationship between {x_feature} and {y_feature}',
            trendline='ols',
            hover_data=['timestamp', 'alarm_type', 'site_id', 'severity']
        )

        st.plotly_chart(fig, use_container_width=True)

        # Temperature vs. alarms analysis if temperature is available
        if 'temperature' in numeric_cols:
            st.markdown("### Temperature vs. Alarms Analysis")

            # Count alarms by temperature
            temp_df = filtered_df.copy()

            # Bin temperature
            temp_df['temp_bin'] = pd.cut(
                temp_df['temperature'],
                bins=10,
                precision=1
            )

            # Count alarms by temperature bin
            temp_counts = temp_df.groupby('temp_bin').size().reset_index(name='count')

            # Create bar chart
            fig = px.bar(
                temp_counts,
                x='temp_bin',
                y='count',
                title='Alarms by Temperature Range',
                color='count',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                xaxis_title='Temperature Range (°C)',
                yaxis_title='Number of Alarms'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Temperature alarms by site
            st.markdown("#### Temperature Distribution by Site")

            # Box plot of temperature by site
            fig = px.box(
                filtered_df,
                x='site_id',
                y='temperature',
                title='Temperature Distribution by Site',
                color='site_id'
            )

            st.plotly_chart(fig, use_container_width=True)

else:  # Custom Dashboard
    st.markdown("## 📊 Custom Dashboard")

    st.markdown("""
    Build your own dashboard by selecting the visualizations you want to see.
    This allows you to create a customized view of the alarm data based on your specific needs.
    """)

    # Create visualization selection
    st.markdown("### Select Visualizations")

    # Create checkboxes for different visualizations
    col1, col2 = st.columns(2)

    with col1:
        show_hotspot = st.checkbox("Network Hotspot Map", value=True)
        show_trends = st.checkbox("Alarm Trends", value=True)
        show_severity = st.checkbox("Severity Distribution", value=True)

    with col2:
        show_heatmap = st.checkbox("Site-Type Heatmap", value=False)
        show_hourly = st.checkbox("Hourly Distribution", value=False)
        show_calendar = st.checkbox("Calendar View", value=False)

    # Display selected visualizations
    dashboard_cols = st.columns(2)

    # Counter for column assignment
    viz_count = 0

    # Hotspot map
    if show_hotspot:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Network Hotspot Map")
            fig = plot_hotspot_map(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # Alarm trends
    if show_trends:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Alarm Trends")
            fig = plot_alarm_trends(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # Severity distribution
    if show_severity:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Severity Distribution")
            fig = plot_alarm_severity(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # Site-type heatmap
    if show_heatmap:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Site-Type Heatmap")
            fig = plot_site_heatmap(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # Hourly distribution
    if show_hourly:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Hourly Distribution")

            # Extract hour
            hour_df = filtered_df.copy()
            hour_df['hour'] = hour_df['timestamp'].dt.hour

            # Count alarms by hour
            hourly_counts = hour_df.groupby('hour').size().reset_index(name='count')

            # Create hour plot
            fig = px.bar(
                hourly_counts,
                x='hour',
                y='count',
                title='Alarms by Hour of Day',
                color='count',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # Calendar view
    if show_calendar:
        with dashboard_cols[viz_count % 2]:
            st.markdown("### Calendar View")
            fig = plot_alarm_calendar(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        viz_count += 1

    # If no visualizations selected
    if viz_count == 0:
        st.info("Please select at least one visualization to display.")

# Key metrics section
st.markdown("---")
st.markdown("## 📈 Key Metrics")

# Create metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_alarms = filtered_df.shape[0]
    st.metric("Total Alarms", total_alarms)

with col2:
    unique_sites = filtered_df['site_id'].nunique()
    st.metric("Unique Sites", unique_sites)

with col3:
    if 'severity' in filtered_df.columns:
        critical_count = filtered_df[filtered_df['severity'] == 'Critical'].shape[0]
        critical_pct = (critical_count / total_alarms) * 100 if total_alarms > 0 else 0
        st.metric("Critical Alarms", f"{critical_count} ({critical_pct:.1f}%)")

with col4:
    if 'timestamp' in filtered_df.columns:
        date_range = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days
        alarms_per_day = total_alarms / max(1, date_range)
        st.metric("Alarms Per Day", f"{alarms_per_day:.1f}")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Predictions](/pages/4_predictions.py)")

with col2:
    st.markdown("[Continue to Insights →](/pages/6_insights.py)")