# 🔮 Predictive Alarm System for Optical Networks

## 📌 Problem Statement

**Prediction of the next event based on the historical network alarm and current alarm**

In large-scale optical communication networks, alarms indicate faults or anomalies—ranging from transient link failures to hardware issues. This project aims to build an intelligent system that analyzes historical and real-time alarms to **predict future alarm occurrences**, **classify their severity**, and **suggest preventive measures**.

> 🚀 Theme: AI/ML | Open Innovation  
> 📅 Submitted for: Nokia Hackathon  

---

## 📊 Why It Matters

Currently, network alarms are handled reactively. Our goal is to enable **proactive or predictive maintenance**, which can:
- Prevent costly downtime
- Reduce SLA violations
- Improve customer satisfaction

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python)
- **Backend**: Python, REST APIs
- **Machine Learning**: Custom models (No pre-trained models used)
- **Visualization**: Matplotlib, Seaborn, Plotly

---

## ✅ Mandatory Tasks Completed

- [x] Ingested and cleaned historical alarm dataset
- [x] Identified patterns between alarms and network conditions
- [x] Developed a machine learning model for alarm prediction

---

## 🌟 Good-to-Have Features

- [x] Alarm severity classification and prioritization
- [x] Root cause probability per alarm category
- [x] Visualization of alarm trends, hotspot regions, and timelines

---

## 🎁 Bonus Tasks Implemented

- [x] Streamlit dashboard showcasing visual insights
- [x] Preventive action suggestions based on model output

---


## 🧠 Future Work

- Real-time simulation using synthetic alarms
- Auto-ticketing integration for NOC teams
- LLM-based root cause extraction from textual logs

---

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Vishnu30543/Nokia_Hackathon_t1.git
cd Nokia_Hackathon_t1
```
2. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the Project
Navigate to the project directory and launch the Streamlit app:
```bash
cd Alarm_dash/project-sb1-hijruqyw/project
streamlit run app.py
```
## 🧑‍💻 Usage

- **Upload Data**  
  Start by uploading your alarm dataset in CSV format or use the sample data provided in the repository.

- **Explore Visualizations**  
  Navigate to the **Visualizations** tab to view:
  - Alarm hotspot maps
  - Temporal patterns
  - Alarm severity distributions

- **Analyze Insights**  
  Go to the **Insights** tab for:
  - Root cause analysis
  - Related alarm correlations
  - Prediction-based summaries

- **Take Action**  
  Review the system’s recommended preventive actions to avoid potential failures and service disruptions.

---

## 👥 Team

**Made with ❤️ by Team-1 – Nokia_Herin_Tech**

