
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/Kingsley-Opoku-Boateng/CMP-Assessment/main/air_quality_data.csv"  # Use your dataset link
    df = pd.read_csv(url)
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Air Quality Data Analysis")
app_mode = st.sidebar.radio("Choose a page", ["Data Overview", "Exploratory Data Analysis", "Modeling and Prediction"])

if app_mode == "Data Overview":
    st.title("Data Overview")
    st.write("### Dataset Information")
    st.write(df.info())  # Show dataset information
    st.write("### First Few Rows of Data")
    st.write(df.head())  # Show first few rows

elif app_mode == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")

    # Basic statistics
    st.write("### Dataset Summary")
    st.write(df.describe())

    # Histograms of numerical columns
    st.write("### Histograms of Numerical Features")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        st.write(f"#### {col}")
        st.bar_chart(df[col])

    # Correlation Matrix
    st.write("### Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

elif app_mode == "Modeling and Prediction":
    st.title("Modeling and Prediction")

    # Select Features and Target
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    target = 'AQI_Bucket'

    # Data Preprocessing
    X = df[features]
    y = df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Create Random Forest model pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    st.write("Training the Random Forest model...")
    rf_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = rf_pipeline.predict(X_test)

    # Evaluation
    st.write("### Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("### Classification Report")
    st.write(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance")
    st.write(feature_importance_df)

    # Predictions on user input
    st.write("### Make a Prediction")
    user_input = {col: st.number_input(f'Enter value for {col}', min_value=float(df[col].min()), max_value=float(df[col].max())) for col in features}
    user_input_df = pd.DataFrame([user_input])
    prediction = rf_pipeline.predict(user_input_df)
    st.write(f"Predicted AQI Bucket: {prediction[0]}")

