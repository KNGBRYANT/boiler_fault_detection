# 🚢 Boiler Fault Detection Dashboard

A **machine learning-powered dashboard** for monitoring boiler performance, detecting early faults, and supporting predictive maintenance — built with **Streamlit, Python, Pandas, and Scikit-Learn**.

This project simulates real-world **marine engineering boiler monitoring** using sensor readings and predictive analytics to classify boilers as **Safe** or **Unsafe**. It also provides interactive data visualization and real-time predictions.

---

## 📖 Project Overview

Boiler systems are critical for ship operations and industrial plants. Even minor faults — such as tube leaks, fouling, or abnormal pressure — can lead to **efficiency loss** or **catastrophic failure**.

This dashboard allows:

- **Real-time monitoring** of boiler parameters  
- **Automatic fault prediction** using trained ML models  
- **Interactive analysis** of historical data  
- **User input testing** to check boiler safety  

---

## ⚙ Features

- **Dataset Overview** – Explore boiler readings with adjustable row display  
- **Performance Metrics** – Average boiler readings and percentage of safe operations  
- **Visual Analytics**:  
  - Safe vs Unsafe count plot  
  - Correlation heatmap of features  
  - Feature distributions  
  - Boxplots comparing Safe vs Unsafe status  
- **Fault Prediction Tool** – Input new readings and instantly get a "Safe" or "Unsafe" prediction  
- **Insights Section** – Highlights key operational takeaways  

---

## 🛠 Technologies Used

- **Python** – Core programming  
- **Pandas / NumPy** – Data handling & preprocessing  
- **Matplotlib / Seaborn** – Data visualization  
- **Scikit-Learn** – Machine learning model training & evaluation  
- **Streamlit** – Interactive dashboard UI  
- **Joblib** – Model saving & loading  


---

## 📊 Dataset Description

| Feature                   | Description                          |
|----------------------------|--------------------------------------|
| **FlueGasTemp**           | Temperature of flue gas (°C)         |
| **SteamPressure**         | Pressure of steam (bar)              |
| **FeedwaterConductivity** | Conductivity of feedwater (µS/cm)    |
| **DrumLevel**             | Water level in boiler drum (%)       |
| **FuelFlow**              | Fuel consumption rate (kg/s)         |
| **OxygenContent**         | Oxygen concentration in flue gas (%) |
| **Status**                | Target label (`Safe` / `Unsafe`)     |

The dataset contains **both safe and unsafe scenarios** to simulate real-world variation.

---

## 📈 Machine Learning

We trained and compared multiple models:

- **Logistic Regression** – Baseline model  
- **Random Forest Classifier** – Best performance (CV Accuracy ≈ 1.00)  
- **Gradient Boosting Classifier** – High accuracy, slightly lower than Random Forest  

**Why Random Forest:**  

- High precision & recall for both Safe and Unsafe classes  
- Robust to outliers and non-linear feature interactions  

---

## 🚀 How to Run the Dashboard Locally

**Clone the repository**

```bash
git clone https://github.com/your-username/boiler-fault-detection.git
cd boiler-fault-detection

pip install -r requirements.txt

streamlit run boiler.py
```
📌 Insights from the Dashboard
-Oxygen Content and Steam Pressure are strong indicators of unsafe boiler operation

-High Feedwater Conductivity often correlates with unsafe conditions

-Unsafe cases usually show multiple extreme parameter values

-Predictive maintenance reduces risk by catching unsafe readings early

💡 Future Improvements
-Connect to real-time sensor data via API

-Add alert notifications when unsafe predictions occur

-Improve model generalization with more diverse real-world data

👨‍🔧 Author
Lawal Mayowa Bryant
Marine Engineer • AI/ML Data Analyst • Prompt Engineer

📧 [Lawalmayoa95@gmail.com]





