import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score


st.markdown(
    """
    <style>
    body {
        background-image: url("https://img.freepik.com/free-vector/illustrated-coronavirus-waste-ocean-background_23-2148743412.jpg");
        background-size: cover;
        font-family: 'Arial', sans-serif;
        color: white;
    }
    .stApp {
        background: rgba(0, 0, 0, 0);
        padding: 20px;
        border-radius: 10px;
    }
    .stTitle, .stHeader, .stMarkdown {
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: rgba(64, 192, 224, 0.87);
        color: black;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Water Quality Prediction System")
st.markdown("## Upload CSV & Train Models")
st.sidebar.header("DATA VISUAL")
# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    # Check if necessary columns exist
    required_columns = {"tds", "turbidity", "potability"}
    if not required_columns.issubset(df.columns):
        st.error("CSV must contain 'tds', 'turbidity', and 'potability' columns!")
    else:
        st.success("File uploaded successfully! ✅")
        st.write(df.head())
        if df.isnull().values.any():
            df.fillna(df.mean(),inplace=True)
        #sidebar headmap
        if st.sidebar.checkbox("Show Correlation Heatmap"):
            st.sidebar.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.sidebar.pyplot(fig)
        # Sidebar checkbox for scatter plot
        if st.sidebar.checkbox("Show Scatter Plot"):
            st.sidebar.markdown("### Scatter Plot of TDS vs Turbidity")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.scatterplot(x=df["tds"], y=df["turbidity"], hue=df["potability"], palette="coolwarm", ax=ax)
            ax.set_xlabel("Total Dissolved Solids (TDS)")
            ax.set_ylabel("Turbidity (NTU)")
            st.sidebar.pyplot(fig)
        # Sidebar checkbox for histogram
        if st.sidebar.checkbox("Show Histogram"):
            st.sidebar.markdown("### Histogram of TDS & Turbidity")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df["tds"], kde=True, color="blue", label="TDS", ax=ax)
            sns.histplot(df["turbidity"], kde=True, color="red", label="Turbidity", ax=ax)
            ax.legend()
            st.sidebar.pyplot(fig)
        #anomaly detection
        st.sidebar.markdown("## 🔍 Anomaly Detection")
        features = st.sidebar.multiselect("Select Features for Anomaly Detection", df.columns.tolist(), default=df.columns.tolist())
        if features:
            data = df[features]
            # Train Isolation Forest Model
            contamination_rate = st.sidebar.slider("⚠ Select Contamination Rate (%)", 1, 10, 2) / 100
            model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42) 
            # Scale Data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            # Get predictions
            predictions = model.fit_predict(scaled_data)
            # Add anomaly labels to data
            data['Anomaly'] = predictions
            data['Anomaly'] = data['Anomaly'].apply(lambda x: "Normal" if x == 1 else "Anomaly")
            # Visualization Only
            st.sidebar.write("### 📉 Anomaly Visualization")
            for feature in features:
                fig, ax = plt.subplots()
                ax.scatter(data[feature], range(len(data)), c=(data['Anomaly'] == "Anomaly"), cmap='coolwarm', label="Anomalies")
                ax.set_xlabel(feature)
                ax.set_ylabel("Index")
                ax.legend()
                st.sidebar.pyplot(fig)
        # Split data
        X = df[["tds","turbidity"]]
        y = df["potability"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        log_model = LogisticRegression()
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        log_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        # Evaluate models
        log_pred = log_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        log_acc = accuracy_score(y_test, log_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        st.write(f"**Logistic Regression Accuracy:** {log_acc:.2f}")
        st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")
        # Prediction Section
        st.markdown("## Make a Prediction")
        tds = st.number_input("Total Dissolved Solids (TDS) in ppm", min_value=0.0, step=0.1, format="%.2f")
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1, format="%.2f")
        model_choice = st.radio("Choose a prediction model:", ["Logistic Regression", "Random Forest"])

        if st.button("Predict"):
            input_data = np.array([[tds, turbidity]])
            input_scaled = scaler.transform(input_data)

            if model_choice == "Logistic Regression":
                prediction = log_model.predict(input_scaled)[0]
            else:
                prediction = rf_model.predict(input_scaled)[0]

            if prediction == 1:
                st.success("The water quality is **Safe** ✅")
            else:
                st.error("The water quality is **Unsafe** ❌")
    
        
