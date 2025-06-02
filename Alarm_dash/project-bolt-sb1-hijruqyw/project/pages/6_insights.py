import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(page_title="Insights & Root Cause Analysis", page_icon="💡", layout="wide")

# Import custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("No data available. Please upload or load sample data from the Data Upload page.")
    st.markdown("[← Go to Data Upload](/pages/1_data_upload.py)")
    st.stop()

# Page header
st.markdown(
    """
    <div class="header">
        <div class="logo">💡</div>
        <div class="title-container">
            <h1 class="main-title">Insights & Root Cause Analysis</h1>
            <p class="subtitle">Discover insights and potential root causes of network alarms</p>
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

# Main content
st.markdown("## 🔍 Root Cause Analysis")

# Selection options
analysis_type = st.radio(
    "Analysis Type",
    ["Alarm Pattern Analysis", "Site-Specific Analysis", "Severity Root Cause"],
    horizontal=True
)

if analysis_type == "Alarm Pattern Analysis":
    # Alarm type selection
    alarm_types = df['alarm_type'].unique()
    selected_alarm = st.selectbox("Select Alarm Type to Analyze", alarm_types)
    
    # Filter data for selected alarm
    alarm_df = df[df['alarm_type'] == selected_alarm]
    
    # Display basic stats
    st.markdown(f"### Analysis of '{selected_alarm}' Alarms")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alarms = alarm_df.shape[0]
        all_alarms = df.shape[0]
        alarm_percentage = (total_alarms / all_alarms) * 100 if all_alarms > 0 else 0
        st.metric("Total Occurrences", f"{total_alarms} ({alarm_percentage:.1f}%)")
    
    with col2:
        unique_sites = alarm_df['site_id'].nunique()
        all_sites = df['site_id'].nunique()
        site_percentage = (unique_sites / all_sites) * 100 if all_sites > 0 else 0
        st.metric("Affected Sites", f"{unique_sites} ({site_percentage:.1f}%)")
    
    with col3:
        if 'severity' in alarm_df.columns:
            severity_counts = alarm_df['severity'].value_counts()
            most_common_severity = severity_counts.index[0] if not severity_counts.empty else "N/A"
            severity_count = severity_counts.iloc[0] if not severity_counts.empty else 0
            severity_percentage = (severity_count / total_alarms) * 100 if total_alarms > 0 else 0
            st.metric("Most Common Severity", f"{most_common_severity} ({severity_percentage:.1f}%)")
    
    with col4:
        if 'timestamp' in alarm_df.columns:
            time_range = (alarm_df['timestamp'].max() - alarm_df['timestamp'].min()).total_seconds() / 3600
            frequency = total_alarms / max(1, time_range) * 24  # Alarms per day
            st.metric("Frequency", f"{frequency:.1f} per day")
    
    # Pattern analysis
    st.markdown("### Temporal Patterns")
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Time Patterns", "Site Distribution"])
    
    with tab1:
        # Time pattern analysis
        hour_df = alarm_df.copy()
        hour_df['hour'] = hour_df['timestamp'].dt.hour
        hour_df['day_of_week'] = hour_df['timestamp'].dt.dayofweek
        
        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hour_df['day_name'] = hour_df['day_of_week'].map(lambda x: day_names[x])
        
        # Count by hour and day of week
        hourly_counts = hour_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
        
        # Create heatmap
        fig = px.density_heatmap(
            hourly_counts,
            x='hour',
            y='day_name',
            z='count',
            title=f'Occurrence Pattern of {selected_alarm} by Day and Hour',
            color_continuous_scale='Viridis'
        )
        
        # Set category order
        fig.update_layout(
            yaxis=dict(
                categoryorder='array',
                categoryarray=day_names
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly pattern
        month_df = alarm_df.copy()
        month_df['month'] = month_df['timestamp'].dt.month
        
        # Map month numbers to names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_df['month_name'] = month_df['month'].map(lambda x: month_names[x-1])
        
        # Count by month
        monthly_counts = month_df.groupby('month_name').size().reset_index(name='count')
        
        # Create bar chart
        fig = px.bar(
            monthly_counts,
            x='month_name',
            y='count',
            title=f'Monthly Distribution of {selected_alarm}',
            color='count',
            color_continuous_scale='Viridis'
        )
        
        # Set category order
        fig.update_layout(
            xaxis=dict(
                categoryorder='array',
                categoryarray=month_names
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Site distribution analysis
        site_counts = alarm_df.groupby('site_id').size().reset_index(name='count')
        site_counts = site_counts.sort_values('count', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            site_counts.head(15),
            x='site_id',
            y='count',
            title=f'Top 15 Sites with {selected_alarm}',
            color='count',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Check for temperature correlation if available
        if 'temperature' in alarm_df.columns:
            # Calculate average temperature for each site
            site_temp = alarm_df.groupby('site_id')['temperature'].mean().reset_index()
            
            # Merge with counts
            site_analysis = pd.merge(site_counts, site_temp, on='site_id')
            
            # Create scatter plot
            fig = px.scatter(
                site_analysis,
                x='temperature',
                y='count',
                title=f'Relationship Between Temperature and {selected_alarm} Frequency',
                color='count',
                hover_data=['site_id'],
                trendline='ols'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = site_analysis['temperature'].corr(site_analysis['count'])
            
            if abs(correlation) > 0.3:
                st.info(f"There appears to be a {'positive' if correlation > 0 else 'negative'} correlation ({correlation:.2f}) between temperature and the frequency of {selected_alarm} alarms.")
    
    # Root cause analysis
    st.markdown("### Potential Root Causes")
    
    # Create mock root cause analysis (in a real system, this would use more sophisticated analysis)
    # Map common root causes for different alarm types
    root_causes = {
        "Link Down": [
            {"cause": "Physical Cable Damage", "probability": 0.35},
            {"cause": "Network Device Failure", "probability": 0.25},
            {"cause": "Power Outage", "probability": 0.20},
            {"cause": "Configuration Error", "probability": 0.15},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "High CPU": [
            {"cause": "Traffic Spike", "probability": 0.40},
            {"cause": "Resource-Intensive Process", "probability": 0.30},
            {"cause": "Insufficient Resources", "probability": 0.15},
            {"cause": "Software Bug", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Memory Overflow": [
            {"cause": "Memory Leak", "probability": 0.45},
            {"cause": "Insufficient Resources", "probability": 0.25},
            {"cause": "Software Bug", "probability": 0.20},
            {"cause": "Configuration Error", "probability": 0.05},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Power Issue": [
            {"cause": "Power Supply Failure", "probability": 0.30},
            {"cause": "External Power Outage", "probability": 0.25},
            {"cause": "UPS Failure", "probability": 0.20},
            {"cause": "Power Fluctuation", "probability": 0.15},
            {"cause": "Other/Unknown", "probability": 0.10}
        ],
        "Temperature Alert": [
            {"cause": "Cooling System Failure", "probability": 0.40},
            {"cause": "High Environmental Temperature", "probability": 0.30},
            {"cause": "Airflow Obstruction", "probability": 0.15},
            {"cause": "Equipment Overload", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Packet Loss": [
            {"cause": "Network Congestion", "probability": 0.35},
            {"cause": "Physical Interface Issues", "probability": 0.25},
            {"cause": "Routing Problems", "probability": 0.20},
            {"cause": "QoS Configuration", "probability": 0.15},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Latency Spike": [
            {"cause": "Network Congestion", "probability": 0.40},
            {"cause": "Routing Changes", "probability": 0.25},
            {"cause": "Resource Contention", "probability": 0.20},
            {"cause": "External Provider Issues", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Authentication Failure": [
            {"cause": "Invalid Credentials", "probability": 0.35},
            {"cause": "Authentication Server Issues", "probability": 0.30},
            {"cause": "Configuration Change", "probability": 0.20},
            {"cause": "Potential Security Incident", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Config Change": [
            {"cause": "Planned Maintenance", "probability": 0.45},
            {"cause": "Unplanned Change", "probability": 0.25},
            {"cause": "Automated Update", "probability": 0.15},
            {"cause": "Unauthorized Access", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ],
        "Interface Flapping": [
            {"cause": "Unstable Physical Connection", "probability": 0.40},
            {"cause": "Duplex Mismatch", "probability": 0.25},
            {"cause": "Hardware Failure", "probability": 0.20},
            {"cause": "Spanning Tree Issues", "probability": 0.10},
            {"cause": "Other/Unknown", "probability": 0.05}
        ]
    }
    
    # Get root causes for selected alarm
    selected_causes = root_causes.get(selected_alarm, [
        {"cause": "Unknown Cause 1", "probability": 0.40},
        {"cause": "Unknown Cause 2", "probability": 0.30},
        {"cause": "Unknown Cause 3", "probability": 0.20},
        {"cause": "Other Issues", "probability": 0.10}
    ])
    
    # Create DataFrame
    causes_df = pd.DataFrame(selected_causes)
    
    # Create pie chart
    fig = px.pie(
        causes_df,
        names='cause',
        values='probability',
        title=f'Probable Root Causes for {selected_alarm}',
        hover_data=['probability']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommended actions
    st.markdown("### Recommended Actions")
    
    # Map recommended actions for different alarm types
    actions_map = {
        "Link Down": [
            "Inspect physical connections and cable integrity",
            "Check power status of connected devices",
            "Verify interface configuration on both ends",
            "Test alternate paths or redundant links",
            "Check for recent configuration changes"
        ],
        "High CPU": [
            "Identify and limit resource-intensive processes",
            "Check for traffic anomalies or DDoS attacks",
            "Increase CPU resources if consistently high",
            "Update device firmware/software if available",
            "Review QoS and traffic shaping configurations"
        ],
        "Memory Overflow": [
            "Restart services showing memory leaks",
            "Increase available memory if possible",
            "Apply vendor patches for known memory issues",
            "Implement memory usage monitoring",
            "Review application configurations"
        ],
        "Power Issue": [
            "Check power supply and connections",
            "Verify UPS functionality and battery status",
            "Inspect for power fluctuations in the facility",
            "Test redundant power supplies if available",
            "Review power management configuration"
        ],
        "Temperature Alert": [
            "Check cooling system functionality",
            "Ensure proper airflow in equipment rooms",
            "Reduce load on affected equipment temporarily",
            "Verify temperature sensor calibration",
            "Implement additional cooling if needed"
        ],
        "Packet Loss": [
            "Check for network congestion and bottlenecks",
            "Inspect physical interfaces for errors",
            "Review routing and switching configurations",
            "Test with different QoS settings",
            "Verify WAN link quality with providers"
        ],
        "Latency Spike": [
            "Identify bandwidth-heavy applications or users",
            "Check for routing inefficiencies",
            "Monitor for microbursts or traffic spikes",
            "Review QoS configurations",
            "Test alternate network paths"
        ],
        "Authentication Failure": [
            "Verify credential validity and user accounts",
            "Check authentication server health",
            "Review recent security policy changes",
            "Look for patterns suggesting brute force attempts",
            "Ensure correct time synchronization"
        ],
        "Config Change": [
            "Review change management logs",
            "Verify configuration compliance with standards",
            "Check for unauthorized access to management systems",
            "Validate configuration backups are available",
            "Test system functionality after changes"
        ],
        "Interface Flapping": [
            "Check for loose or damaged physical connections",
            "Verify interface error counters for clues",
            "Test for duplex mismatches",
            "Review spanning tree configuration",
            "Replace interface hardware if persistent"
        ]
    }
    
    # Get actions for selected alarm
    selected_actions = actions_map.get(selected_alarm, [
        "Investigate the root cause",
        "Monitor for recurring issues",
        "Check related systems",
        "Document findings for future reference"
    ])
    
    # Display actions
    for i, action in enumerate(selected_actions, 1):
        st.markdown(f"{i}. {action}")
    
    # Related alarms analysis
    st.markdown("### Related Alarms Analysis")
    
    # Find alarms that frequently occur before or after the selected alarm
    # For demonstration, we'll create a mock relationship
    
    # Get unique alarm types excluding the selected one
    other_alarms = [a for a in alarm_types if a != selected_alarm]
    
    # Create mock relationships (in a real system, this would use sequence analysis)
    precursor_alarms = random.sample(other_alarms, min(3, len(other_alarms)))
    successor_alarms = random.sample(other_alarms, min(3, len(other_alarms)))
    
    # Create mock probabilities
    precursor_probs = [round(random.uniform(0.1, 0.9), 2) for _ in range(len(precursor_alarms))]
    successor_probs = [round(random.uniform(0.1, 0.9), 2) for _ in range(len(successor_alarms))]
    
    # Normalize probabilities
    precursor_probs = [p / sum(precursor_probs) for p in precursor_probs]
    successor_probs = [p / sum(successor_probs) for p in successor_probs]
    
    # Create DataFrames
    precursor_df = pd.DataFrame({
        'alarm_type': precursor_alarms,
        'probability': precursor_probs
    })
    
    successor_df = pd.DataFrame({
        'alarm_type': successor_alarms,
        'probability': successor_probs
    })
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Precursor Alarms")
        st.markdown("Alarms that frequently occur before this alarm:")
        
        # Create bar chart
        fig = px.bar(
            precursor_df,
            x='alarm_type',
            y='probability',
            title='Alarms that Precede ' + selected_alarm,
            color='probability',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Successor Alarms")
        st.markdown("Alarms that frequently occur after this alarm:")
        
        # Create bar chart
        fig = px.bar(
            successor_df,
            x='alarm_type',
            y='probability',
            title='Alarms that Follow ' + selected_alarm,
            color='probability',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Site-Specific Analysis":
    # Site selection
    sites = df['site_id'].unique()
    selected_site = st.selectbox("Select Site to Analyze", sites)
    
    # Filter data for selected site
    site_df = df[df['site_id'] == selected_site]
    
    # Display basic stats
    st.markdown(f"### Analysis of Site: {selected_site}")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alarms = site_df.shape[0]
        all_alarms = df.shape[0]
        site_percentage = (total_alarms / all_alarms) * 100 if all_alarms > 0 else 0
        st.metric("Total Alarms", f"{total_alarms} ({site_percentage:.1f}%)")
    
    with col2:
        unique_types = site_df['alarm_type'].nunique()
        all_types = df['alarm_type'].nunique()
        type_percentage = (unique_types / all_types) * 100 if all_types > 0 else 0
        st.metric("Alarm Types", f"{unique_types} ({type_percentage:.1f}%)")
    
    with col3:
        if 'severity' in site_df.columns:
            critical_count = site_df[site_df['severity'] == 'Critical'].shape[0]
            critical_percentage = (critical_count / total_alarms) * 100 if total_alarms > 0 else 0
            st.metric("Critical Alarms", f"{critical_count} ({critical_percentage:.1f}%)")
    
    with col4:
        if 'timestamp' in site_df.columns:
            time_range = (site_df['timestamp'].max() - site_df['timestamp'].min()).total_seconds() / 3600
            frequency = total_alarms / max(1, time_range) * 24  # Alarms per day
            st.metric("Frequency", f"{frequency:.1f} per day")
    
    # Alarm distribution
    st.markdown("### Alarm Distribution")
    
    # Count alarms by type
    type_counts = site_df.groupby('alarm_type').size().reset_index(name='count')
    type_counts = type_counts.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        type_counts,
        x='alarm_type',
        y='count',
        title=f'Alarm Types at {selected_site}',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Severity distribution
    st.markdown("### Severity Distribution")
    
    # Count alarms by severity
    severity_counts = site_df.groupby('severity').size().reset_index(name='count')
    
    # Create pie chart
    fig = px.pie(
        severity_counts,
        names='severity',
        values='count',
        title=f'Alarm Severity at {selected_site}',
        color='severity',
        color_discrete_map={
            'Critical': '#FF5252',
            'Major': '#FF9800',
            'Minor': '#FFC107',
            'Warning': '#8BC34A',
            'Info': '#2196F3'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.markdown("### Temporal Analysis")
    
    # Create tabs for different time analyses
    tab1, tab2 = st.tabs(["Alarm Trends", "Time Patterns"])
    
    with tab1:
        # Resample by day
        daily_df = site_df.copy()
        daily_df['date'] = daily_df['timestamp'].dt.date
        
        # Count alarms by date and type
        daily_counts = daily_df.groupby(['date', 'alarm_type']).size().reset_index(name='count')
        
        # Create line plot
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            color='alarm_type',
            title=f'Alarm Trends at {selected_site}',
            line_shape='spline'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Extract hour and day of week
        time_df = site_df.copy()
        time_df['hour'] = time_df['timestamp'].dt.hour
        time_df['day_of_week'] = time_df['timestamp'].dt.dayofweek
        
        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_df['day_name'] = time_df['day_of_week'].map(lambda x: day_names[x])
        
        # Count alarms by hour and day
        time_counts = time_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
        
        # Create heatmap
        fig = px.density_heatmap(
            time_counts,
            x='hour',
            y='day_name',
            z='count',
            title=f'Alarm Patterns at {selected_site}',
            color_continuous_scale='Viridis'
        )
        
        # Set category order
        fig.update_layout(
            yaxis=dict(
                categoryorder='array',
                categoryarray=day_names
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Site health score
    st.markdown("### Site Health Analysis")
    
    # Calculate health score based on alarms (mock calculation for demo)
    # In a real system, this would be a more sophisticated calculation
    
    # Severity weights
    severity_weights = {
        'Critical': 10,
        'Major': 5,
        'Minor': 2,
        'Warning': 1,
        'Info': 0.5
    }
    
    # Calculate weighted score
    if 'severity' in site_df.columns:
        site_df['weight'] = site_df['severity'].map(lambda x: severity_weights.get(x, 1))
        total_weight = site_df['weight'].sum()
        
        # Normalize to 0-100 scale (higher is worse)
        health_score = 100 - min(100, (total_weight / total_alarms) * 10) if total_alarms > 0 else 100
        
        # Display gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': f"Site Health Score - {selected_site}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Health interpretation
        if health_score >= 80:
            st.success(f"Site {selected_site} is in good health with minimal critical issues.")
        elif health_score >= 50:
            st.warning(f"Site {selected_site} is experiencing moderate issues that should be addressed.")
        else:
            st.error(f"Site {selected_site} is in poor health and requires immediate attention!")
    
    # Recommendations
    st.markdown("### Site-Specific Recommendations")
    
    # Get most common alarm types
    common_alarms = type_counts.head(3)['alarm_type'].tolist()
    
    # Create recommendations based on common alarms
    recommendations = []
    
    # Map of recommendations by alarm type
    recommendations_map = {
        "Link Down": "Check physical connectivity and power status at this site.",
        "High CPU": "Review resource usage and consider upgrading hardware at this site.",
        "Memory Overflow": "Investigate memory leaks and optimize application configurations.",
        "Power Issue": "Inspect power supply systems and ensure proper UPS functionality.",
        "Temperature Alert": "Check cooling systems and ensure proper airflow in the equipment room.",
        "Packet Loss": "Investigate network congestion and check for physical interface issues.",
        "Latency Spike": "Review bandwidth usage patterns and check for routing inefficiencies.",
        "Authentication Failure": "Verify credential configurations and check authentication servers.",
        "Config Change": "Review change management procedures for this site.",
        "Interface Flapping": "Check for unstable physical connections and duplex mismatches."
    }
    
    # Add recommendations for common alarms
    for alarm in common_alarms:
        if alarm in recommendations_map:
            recommendations.append(recommendations_map[alarm])
    
    # Add general recommendations
    if health_score < 70:
        recommendations.append("Consider a comprehensive health check of all equipment at this site.")
    
    if 'temperature' in site_df.columns and site_df['temperature'].mean() > 28:
        recommendations.append("Monitor environmental conditions as temperatures appear elevated.")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

else:  # Severity Root Cause
    # Severity selection
    severities = df['severity'].unique()
    selected_severity = st.selectbox("Select Severity Level to Analyze", severities)
    
    # Filter data for selected severity
    severity_df = df[df['severity'] == selected_severity]
    
    # Display basic stats
    st.markdown(f"### Analysis of '{selected_severity}' Alarms")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alarms = severity_df.shape[0]
        all_alarms = df.shape[0]
        severity_percentage = (total_alarms / all_alarms) * 100 if all_alarms > 0 else 0
        st.metric("Total Occurrences", f"{total_alarms} ({severity_percentage:.1f}%)")
    
    with col2:
        unique_types = severity_df['alarm_type'].nunique()
        all_types = df['alarm_type'].nunique()
        type_percentage = (unique_types / all_types) * 100 if all_types > 0 else 0
        st.metric("Alarm Types", f"{unique_types} ({type_percentage:.1f}%)")
    
    with col3:
        unique_sites = severity_df['site_id'].nunique()
        all_sites = df['site_id'].nunique()
        site_percentage = (unique_sites / all_sites) * 100 if all_sites > 0 else 0
        st.metric("Affected Sites", f"{unique_sites} ({site_percentage:.1f}%)")
    
    with col4:
        if 'timestamp' in severity_df.columns:
            time_range = (severity_df['timestamp'].max() - severity_df['timestamp'].min()).total_seconds() / 3600
            frequency = total_alarms / max(1, time_range) * 24  # Alarms per day
            st.metric("Frequency", f"{frequency:.1f} per day")
    
    # Alarm type distribution
    st.markdown("### Alarm Type Distribution")
    
    # Count alarms by type
    type_counts = severity_df.groupby('alarm_type').size().reset_index(name='count')
    type_counts = type_counts.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        type_counts,
        x='alarm_type',
        y='count',
        title=f'Distribution of {selected_severity} Alarms by Type',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Site distribution
    st.markdown("### Site Distribution")
    
    # Count alarms by site
    site_counts = severity_df.groupby('site_id').size().reset_index(name='count')
    site_counts = site_counts.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        site_counts.head(15),
        x='site_id',
        y='count',
        title=f'Top 15 Sites with {selected_severity} Alarms',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.markdown("### Temporal Analysis")
    
    # Resample by day
    daily_df = severity_df.copy()
    daily_df['date'] = daily_df['timestamp'].dt.date
    
    # Count alarms by date
    daily_counts = daily_df.groupby('date').size().reset_index(name='count')
    
    # Create line plot
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        title=f'Trends of {selected_severity} Alarms Over Time',
        line_shape='spline'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Root cause analysis
    st.markdown("### Root Cause Analysis")
    
    # Get most common alarm types for this severity
    common_alarms = type_counts.head(3)['alarm_type'].tolist()
    
    # Create mock root causes based on common alarms and severity
    causes = []
    
    # Add causes for common alarms
    for alarm in common_alarms:
        causes.append({
            "cause": f"{alarm} issues",
            "description": f"Problems related to {alarm} events",
            "probability": round(random.uniform(0.1, 0.3), 2)
        })
    
    # Add severity-specific causes
    if selected_severity == "Critical":
        causes.append({
            "cause": "Hardware Failures",
            "description": "Critical hardware component failures",
            "probability": 0.25
        })
        causes.append({
            "cause": "Network Outages",
            "description": "Complete loss of network connectivity",
            "probability": 0.20
        })
    elif selected_severity == "Major":
        causes.append({
            "cause": "Performance Degradation",
            "description": "Significant reduction in system performance",
            "probability": 0.25
        })
        causes.append({
            "cause": "Partial Service Disruption",
            "description": "Some services unavailable or degraded",
            "probability": 0.20
        })
    elif selected_severity == "Minor":
        causes.append({
            "cause": "Configuration Issues",
            "description": "Misconfigurations causing minor problems",
            "probability": 0.25
        })
        causes.append({
            "cause": "Resource Limitations",
            "description": "Systems approaching resource limits",
            "probability": 0.20
        })
    else:
        causes.append({
            "cause": "Informational Events",
            "description": "Normal system operations and changes",
            "probability": 0.25
        })
        causes.append({
            "cause": "Transient Issues",
            "description": "Temporary problems that resolve automatically",
            "probability": 0.20
        })
    
    # Create DataFrame
    causes_df = pd.DataFrame(causes)
    
    # Normalize probabilities
    total_prob = causes_df['probability'].sum()
    if total_prob > 0:
        causes_df['probability'] = causes_df['probability'] / total_prob
    
    # Create bar chart
    fig = px.bar(
        causes_df,
        x='cause',
        y='probability',
        title=f'Root Causes of {selected_severity} Alarms',
        color='probability',
        color_continuous_scale='Viridis',
        hover_data=['description']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Impact analysis
    st.markdown("### Impact Analysis")
    
    # Create mock impact metrics based on severity
    impact_metrics = []
    
    if selected_severity == "Critical":
        impact_metrics = [
            {"metric": "Service Availability", "impact": 9.5},
            {"metric": "Customer Experience", "impact": 9.0},
            {"metric": "Revenue Loss", "impact": 8.5},
            {"metric": "Operational Efficiency", "impact": 8.0},
            {"metric": "Compliance Risk", "impact": 7.5}
        ]
    elif selected_severity == "Major":
        impact_metrics = [
            {"metric": "Service Availability", "impact": 7.0},
            {"metric": "Customer Experience", "impact": 7.5},
            {"metric": "Revenue Loss", "impact": 6.0},
            {"metric": "Operational Efficiency", "impact": 6.5},
            {"metric": "Compliance Risk", "impact": 5.5}
        ]
    elif selected_severity == "Minor":
        impact_metrics = [
            {"metric": "Service Availability", "impact": 4.0},
            {"metric": "Customer Experience", "impact": 4.5},
            {"metric": "Revenue Loss", "impact": 3.0},
            {"metric": "Operational Efficiency", "impact": 5.0},
            {"metric": "Compliance Risk", "impact": 3.5}
        ]
    else:
        impact_metrics = [
            {"metric": "Service Availability", "impact": 2.0},
            {"metric": "Customer Experience", "impact": 2.5},
            {"metric": "Revenue Loss", "impact": 1.0},
            {"metric": "Operational Efficiency", "impact": 3.0},
            {"metric": "Compliance Risk", "impact": 1.5}
        ]
    
    # Create DataFrame
    impact_df = pd.DataFrame(impact_metrics)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=impact_df['impact'],
        theta=impact_df['metric'],
        fill='toself',
        name=selected_severity
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title=f'Impact Assessment of {selected_severity} Alarms'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### Recommendations")
    
    # Create recommendations based on severity
    recommendations = []
    
    if selected_severity == "Critical":
        recommendations = [
            "Implement immediate response procedures for critical alarms",
            "Establish clear escalation paths to technical teams and management",
            "Create redundant systems for common critical failure points",
            f"Focus on {common_alarms[0] if common_alarms else 'common alarm types'} which account for most critical issues",
            "Schedule regular preventive maintenance for critical systems"
        ]
    elif selected_severity == "Major":
        recommendations = [
            "Develop standard operating procedures for major alarms",
            "Implement proactive monitoring to catch issues before they escalate",
            "Train support teams on troubleshooting common major issues",
            f"Review systems generating {common_alarms[0] if common_alarms else 'common alarm types'} for potential upgrades",
            "Create recovery playbooks for major incidents"
        ]
    elif selected_severity == "Minor":
        recommendations = [
            "Implement batch processing of minor alarms to reduce noise",
            "Create automated responses for common minor issues",
            "Schedule regular maintenance windows to address accumulated minor issues",
            "Monitor minor alarms for potential patterns indicating larger problems",
            "Document common resolutions for quick reference"
        ]
    else:
        recommendations = [
            "Filter and categorize informational alarms for easier analysis",
            "Use informational alarms to identify potential improvement areas",
            "Create baseline metrics for normal operation based on informational data",
            "Implement regular reviews of informational alarm patterns",
            "Consider reducing unnecessary informational alarms to decrease noise"
        ]
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# Navigation buttons
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("[← Back to Visualizations](/pages/5_visualizations.py)")

with col2:
    st.markdown("[Continue to Architecture →](/pages/7_architecture.py)")