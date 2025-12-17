import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix

# Load your dataset
@st.cache
# Use the raw URL of your dataset from GitHub
def load_data():
    url = "https://raw.githubusercontent.com/KingsleyOpokuBoateng/CMP-Assessment/main/air_quality_data.csv"
    return pd.read_csv(url)

# Main App
def main():
    st.set_page_config(page_title="Air Quality Analysis", page_icon="🌫️", layout="wide")
    
    # Add a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ("Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"))
    
    # Display the selected page
    if page == "Data Overview":
        display_data_overview()
    elif page == "Exploratory Data Analysis (EDA)":
        perform_eda()
    elif page == "Modelling and Prediction":
        train_model()

# Data Overview: Display general information about the dataset
def display_data_overview():
    df = load_data()
    
    st.title("Data Overview")
    st.write("### Dataset Information")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write("### Column Data Types:")
    st.write(df.dtypes)
    st.write("### Missing Data:")
    st.write(df.isnull().sum())
    
    st.write("### Sample Data:")
    st.dataframe(df.head())

# EDA: Generate exploratory data analysis (EDA) visualizations
def perform_eda():
    df = load_data()
    
    st.title("Exploratory Data Analysis (EDA)")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Distribution of pollutants
    st.subheader("Pollutant Distributions")
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    fig, ax = plt.subplots(figsize=(10, 6))
    df[pollutants].hist(bins=30, ax=ax)
    st.pyplot(fig)

    # Boxplot for specific pollutants
    st.subheader("Pollutant Boxplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[pollutants], ax=ax)
    st.pyplot(fig)

# Modelling and Prediction: Build and evaluate machine learning models
def train_model():
    df = load_data()
    st.title("Modelling and Prediction")
    
    # Select features and target variable
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    target = 'AQI_Bucket'
    
    # Data preprocessing (Handle missing values, encode categorical variables, etc.)
    X = df[features]
    y = df[target]
    
    # Handle missing values (imputation)
    X.fillna(X.mean(), inplace=True)  # Replace missing values with column means
    
    # Encoding categorical variables (e.g., City, Season)
    X = pd.get_dummies(X, drop_first=True)
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train a model (Random Forest for simplicity)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    # Feature Importance
    feature_importances = model.feature_importances_
    st.subheader("Feature Importances")
    st.bar_chart(feature_importances)

# Run the app
if __name__ == "__main__":
    main()
