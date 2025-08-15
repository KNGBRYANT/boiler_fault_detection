import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset & model
df = pd.read_csv("boiler_fault_dataset.csv")
model = joblib.load("boiler_fault_model.pkl")


# App title
st.title("🚢 Boiler Fault Detection Dashboard")
st.write("Monitor boiler performance and predict possible faults.")

# ===== Section 1: Overview =====

st.header("📊 Dataset Overview")

# User chooses how many rows to show
num_rows = st.slider("Select number of rows to view", min_value=5, max_value=len(df), value=5, step=5)
st.write(df.head(num_rows))

col1, col2, col3 = st.columns(3)
col1.metric("Avg Flue Gas Temp", f"{df['FlueGasTemp'].mean():.2f} °C")
col2.metric("Avg Steam Pressure", f"{df['SteamPressure'].mean():.2f} bar")
safe_percentage = (df['Status'].value_counts(normalize=True)['Safe'] * 100)
col3.metric("% Safe Boilers", f"{safe_percentage:.1f}%")

# ===== Section 2: Visualizations =====
st.header("📈 Data Visualizations")

# Safe vs Unsafe Count
fig, ax = plt.subplots()
sns.countplot(x="Status", data=df, palette="coolwarm", ax=ax)
ax.set_title("Safe vs Unsafe Boilers")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.drop(columns=["Status"]).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Feature Distributions
st.subheader("Feature Distributions")
for column in df.drop(columns=["Status"]).columns:
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax, color="blue")
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

# 📦 Status vs Features
st.subheader("📦 Status vs Features")

for column in df.drop(columns=["Status"]).columns:
    fig, ax = plt.subplots()
    sns.boxplot(x="Status", y=column, data=df, palette="coolwarm", ax=ax)
    ax.set_title(f"{column} by Boiler Status")
    st.pyplot(fig)



# ===== Section 3: Prediction Tool =====
st.header("🛠 Boiler Status Prediction")

# Input form
with st.form("prediction_form"):
    flue_temp = st.number_input("Flue Gas Temp", min_value=150.0, max_value=400.0, value=250.0)
    steam_pressure = st.number_input("Steam Pressure", min_value=20.0, max_value=100.0, value=55.0)
    conductivity = st.number_input("Feedwater Conductivity", min_value=0.0, max_value=10.0, value=2.5)
    drum_level = st.number_input("Drum Level", min_value=0.0, max_value=120.0, value=60.0)
    fuel_flow = st.number_input("Fuel Flow", min_value=0.0, max_value=15.0, value=5.0)
    oxygen_content = st.number_input("Oxygen Content", min_value=0.0, max_value=10.0, value=3.8)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = [[flue_temp, steam_pressure, conductivity, drum_level, fuel_flow, oxygen_content]]
    prediction = model.predict(input_data)[0]
    if prediction == "Safe":
        st.success("✅ Boiler is operating safely!")
    else:
        st.error("⚠ Warning: Boiler may be unsafe!")

# ===== Section 4: Feature Importance =====
st.header("🔍 Model Feature Importance")

try:
    # Get feature importances from Random Forest model
    importances = model.feature_importances_
    feature_names = df.drop(columns=["Status"]).columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display table
    st.write(importance_df)

    # Plot feature importance
    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm", ax=ax)
    ax.set_title("Feature Importance for Boiler Fault Detection")
    st.pyplot(fig)

except AttributeError:
    st.warning("Feature importance is only available for tree-based models like Random Forest.")

# ===== Section 5: Insights Summary =====
st.header("📝 Insights & Interpretation")

insights_text = """
### 1. Safety Performance
- The *Safe vs Unsafe Boilers* chart shows the overall balance of healthy vs at-risk boilers.
- The percentage of safe boilers is a quick KPI to assess overall plant health.

### 2. Key Risk Indicators
- Certain features, such as **Flue Gas Temperature** and **Steam Pressure**, have a strong influence on whether a boiler is considered safe.
- Large deviations in these readings often align with Unsafe predictions.

### 3. Relationships Between Variables
- The **Status vs Features boxplots** reveal patterns: for example, Unsafe boilers may have higher flue gas temperatures or lower drum levels.
- The correlation heatmap helps identify which sensors move together, useful for maintenance planning.

### 4. Predictive Model
- The prediction tool allows real-time entry of sensor data to check safety status before problems escalate.
- This can help operators take preventive action, reducing downtime and maintenance costs.

### 5. Next Steps
- Regular monitoring of these patterns can improve predictive maintenance schedules.
- Focus maintenance efforts on the top contributing features identified in feature importance.

---

**Bottom line:** This dashboard is both a monitoring tool and a decision-support system. It not only flags potential faults but also guides you on *where* to look and *why* the model thinks a fault might occur.
"""

st.markdown(insights_text)
