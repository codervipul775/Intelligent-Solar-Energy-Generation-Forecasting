# 🌞 Intelligent Solar Energy Generation Forecasting and Agentic Grid Optimization System

## 📌 Project Overview

This project aims to design and implement an **AI-driven solar energy forecasting and grid optimization system**. The system predicts solar power generation using historical and weather-related data and extends this capability into an **agentic AI assistant** that generates structured, explainable recommendations for grid optimization and energy utilization.

The project is developed in **two milestones**:

* **Milestone 1:** Machine Learning–based Solar Energy Forecasting
* **Milestone 2:** Agentic AI Grid Optimization Assistant

The final application is **publicly hosted**, uses **only free-tier/open-source tools**, and provides a **user-friendly web interface**.

---

## 🎯 Problem Statement & Use Case

Solar energy generation is highly variable due to weather conditions and seasonal patterns. Grid operators require accurate forecasts and intelligent decision support to:

* Balance supply and demand
* Reduce energy wastage
* Improve renewable energy integration
* Plan storage and load shifting strategies

This project provides:

* Accurate short-term/long-term solar generation forecasts
* Automated analysis of variability and risk
* AI-generated, structured grid optimization recommendations

---

## 🏗️ Overall System Architecture

### High-Level Architecture

```
User (Browser)
   │
   ▼
Streamlit Web UI
   │
   ├── Data Upload & Selection
   ├── Forecast Horizon Selection
   ├── Visualization (Graphs & Metrics)
   │
   ▼
Backend Analytics Layer (Python)
   │
   ├── Data Preprocessing & Feature Engineering
   ├── ML / Time-Series Forecasting Models
   ├── Model Evaluation (MAE, RMSE)
   │
   ▼
Forecast Outputs (Structured Data)
   │
   ├── Visualization Module
   └── Agentic AI System (Milestone 2)
           │
           ├── Forecast Analysis Agent
           ├── Variability & Risk Detection Agent
           ├── Knowledge Retrieval Agent (RAG)
           └── Optimization Recommendation Agent
                   │
                   ▼
           Structured Grid Optimization Report
```

> 📌 *Note:* The agentic AI system provides **decision-support recommendations** only and does not directly control any grid infrastructure.

---

## 🧩 Milestone 1: ML-Based Solar Energy Forecasting (Mid-Sem)

### Objective

Build a machine learning or time-series forecasting system to predict solar energy generation.

### Inputs

* Historical solar power generation data
* Weather indicators (irradiance, temperature, cloud cover)
* Time-based features (hour, day, month, season)

### Functional Requirements

* Data preprocessing and cleaning
* Feature engineering
* Solar energy forecasting
* Trend and seasonality analysis
* Visualization of predictions

### Technical Requirements

* ML or time-series models such as:

  * Linear Regression
  * Random Forest Regressor
  * ARIMA / SARIMA
  * Prophet (optional)
* Evaluation metrics:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Square Error)

### UI Requirements

* Dataset upload or selection
* Forecast horizon selection
* Line plots (Actual vs Predicted)
* Trend and seasonality visualizations

### Milestone 1 Deliverables

* Problem understanding & use-case description
* Input–output specification
* Forecasting pipeline architecture
* Working application with basic UI
* Forecast accuracy evaluation

---

## 🤖 Milestone 2: Agentic AI Grid Optimization Assistant (End-Sem)

### Objective

Extend the forecasting system into an **agentic AI assistant** that reasons about forecast variability and generates structured grid optimization recommendations.

### Functional Requirements

* Analyze forecast outputs and uncertainty
* Identify variability and risk periods
* Retrieve renewable energy and grid management guidelines
* Generate structured optimization recommendations
* Handle incomplete or uncertain data gracefully

### Technical Requirements

* Open-source or free-tier LLM integration
* Agentic workflow with explicit state management (LangGraph)
* Retrieval-Augmented Generation (optional but recommended)
* Prompt strategies to avoid unsupported claims

### Structured Output Report

The generated report includes:

* Solar generation forecast summary
* Identified variability and risk periods
* Grid balancing and storage recommendations
* Energy utilization optimization strategies
* Supporting references

### Optional Extensions

* Battery storage optimization analysis
* Multi-site solar forecasting
* PDF export of optimization report
* Scenario-based energy planning

---

## 🧪 Dataset

* **Solar Energy Power Generation Dataset**
* Source: Kaggle
* Link: [https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset](https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset)

---

## 🛠️ Tech Stack & Tools

### Core Technologies

| Layer           | Tools / Libraries                        | Use Case                            |
| --------------- | ---------------------------------------- | ----------------------------------- |
| Language        | Python 3.10+                             | Core development                    |
| Data Processing | Pandas, NumPy                            | Data cleaning & feature engineering |
| ML Models       | scikit-learn                             | Regression models                   |
| Time Series     | statsmodels, Prophet                     | Seasonal forecasting                |
| Visualization   | Matplotlib, Seaborn                      | Trend & prediction plots            |
| UI              | Streamlit                                | Interactive web interface           |
| LLMs            | Open-source models (LLaMA, Mistral, Phi) | Recommendation generation           |
| Agent Framework | LangGraph                                | Multi-agent workflows               |
| Vector Store    | FAISS / Chroma                           | Knowledge retrieval (RAG)           |
| Hosting         | Hugging Face Spaces                      | Public deployment                   |

> ❗ No paid APIs are used in this project.

---

## 📁 Repository Structure

```
Intelligent-Solar-Energy-Generation-Forecasting/
├── data/
│   └── solar_generation.csv
├── notebooks/
│   ├── eda.ipynb
│   └── experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── forecast.py
│   ├── evaluate.py
│   └── agents/
│       ├── forecast_agent.py
│       ├── risk_agent.py
│       ├── retrieval_agent.py
│       ├── optimization_agent.py
│       └── agent_graph.py
├── ui/
│   └── app.py
├── vectorstore/
│   └── energy_docs/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Project Setup Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yashsingh045/Intelligent-Solar-Energy-Generation-Forecasting.git

```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\\Scripts\\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application (Local)

```bash
streamlit run ui/app.py
```



---

## 🌐 Deployment

The application is deployed using **Hugging Face Spaces (Streamlit)**:

* Free-tier hosting
* Publicly accessible URL
* Automatic build from repository


---

## 🚀 Future Improvements

* Integration of real-time weather data (free APIs)
* Advanced deep learning models (LSTM, Temporal CNNs)
* Enhanced uncertainty quantification
* Real-world grid simulation scenarios

---

## 👥 Project Contributors

* [Yashveer Singh](https://github.com/Yashsingh045)
* [Abhay Pratap Yadav](https://github.com/QUBITABHAY)
* [Vipul](https://github.com/codervipul775)
* [Ananya Gupta](https://github.com/anya-xcode)


---

✨ *This project demonstrates the intersection of machine learning, renewable energy analytics, and agentic AI systems to support sustainable power grid operations.*
