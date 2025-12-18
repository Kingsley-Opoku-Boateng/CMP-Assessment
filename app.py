import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Set the Streamlit page configuration
st.set_page_config(page_title="Air Quality Prediction App", page_icon=":bar_chart:")

@st.cache_data
def load_data():
    # URL to the dataset 
    url = "https://raw.githubusercontent.com/Kingsley-Opoku-Boateng/CMP-Assessment/main/air_quality_data.csv"
    return pd.read_csv(url)

# Load data into session state
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# -------------------------------------
# Page 1: Data Overview
# -------------------------------------
def data_overview():
    st.title("Data Overview :open_file_folder:")

    st.markdown("""
    **Goal:** Let's get familiar with the dataset and understand its structure.

    - **Dataset Overview**: Display basic data structure
    - **Descriptive Statistics**: Get summary statistics
    - **Missing Values**: Identify any missing data
    """)

    data = st.session_state['data']

    # Dataset Information
    st.header("Dataset Information :eyes:")
    st.write(data.info())

    # Preview data
    st.header("Data Preview :mag:")
    num_rows = st.slider("Select number of rows to preview:", 1, 100, 10)
    st.write(data.head(num_rows))

    # Descriptive Statistics
    st.header("Descriptive Statistics :clipboard:")
    show_desc_table = st.checkbox("Show Descriptive Statistics")
    if show_desc_table:
        st.write(data.describe())

    # Missing values
    st.header("Missing Values Analysis :question:")
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
    st.write(missing_df)

# -------------------------------------
# Page 2: Data Preprocessing
# -------------------------------------
def data_preprocessing():
    st.title("Data Preprocessing :broom:")

    st.markdown("""
    **Goal:** Clean and transform the dataset into a suitable format for analysis.

    - **Handle Missing Data**
    - **Feature Engineering** (Adding AQI, AQI_Bucket)
    - **Drop Unnecessary Columns**
    """)

    data = st.session_state['data']

    # Handle Missing Data
    st.header("Handling Missing Data :umbrella:")
    imputation_method = st.radio("Choose imputation method:", ["Mean", "Median", "Mode"])
    columns_to_impute = st.multiselect("Select columns to impute:", data.columns)

    if st.button("Impute Missing Values"):
        for column in columns_to_impute:
            if imputation_method == "Mean":
                data[column].fillna(data[column].mean(), inplace=True)
            elif imputation_method == "Median":
                data[column].fillna(data[column].median(), inplace=True)
            elif imputation_method == "Mode":
                data[column].fillna(data[column].mode()[0], inplace=True)
        st.success("Missing values imputed successfully.")
        st.session_state['data'] = data

    # Feature Engineering (Adding AQI and AQI Bucket)
    st.header("Feature Engineering :hammer_and_wrench:")
    if st.checkbox("Add AQI and AQI_Bucket columns"):
        def calculate_aqi(row):
            aqi_values = {
                'PM2.5': row['PM2.5'],
                'PM10': row['PM10'],
                'NO2': row['NO2'],
                'CO': row['CO'],
                'O3': row['O3'],
                'SO2': row['SO2']
            }
            return max(aqi_values.values())

        data['AQI'] = data.apply(calculate_aqi, axis=1)

        def get_aqi_bucket(aqi):
            if aqi <= 50:
                return 'Good'
            elif aqi <= 100:
                return 'Moderate'
            elif aqi <= 150:
                return 'Unhealthy for Sensitive Groups'
            elif aqi <= 200:
                return 'Unhealthy'
            elif aqi <= 300:
                return 'Very Unhealthy'
            else:
                return 'Hazardous'

        data['AQI_Bucket'] = data['AQI'].apply(get_aqi_bucket)
        st.success("AQI and AQI_Bucket columns added successfully.")
        st.session_state['data'] = data

    # Drop Unnecessary Columns
    st.header("Dropping Columns :wastebasket:")
    columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
    if st.button("Drop Columns"):
        data.drop(columns=columns_to_drop, inplace=True)
        st.success("Columns dropped successfully.")
        st.session_state['data'] = data

# -------------------------------------
# Page 3: Exploratory Data Analysis (EDA)
# -------------------------------------
def eda():
    st.title("Exploratory Data Analysis (EDA) :bar_chart:")

    st.markdown("""
    **Goal:** Explore the dataset through visualizations.

    - **Pollutant Distribution**
    - **Correlation Matrix**
    - **Pollutant Comparison**
    """)

    data = st.session_state['data']

    # Pollutant Distribution
    st.header("Pollutant Distribution :chart_with_upwards_trend:")
    pollutants = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene']
    for pollutant in pollutants:
        st.subheader(f"{pollutant} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[pollutant], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation Matrix
    st.header("Correlation Matrix :link:")
    corr_matrix = data[['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene'']].corr()
    st.write(corr_matrix)

    # Heatmap of Correlation
    st.header("Correlation Heatmap :flame:")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)

# -------------------------------------
# Page 4: Modeling and Prediction
# -------------------------------------
def modeling():
    st.title("Modeling and Prediction :robot_face:")

    st.markdown("""
    **Goal:** Build and evaluate a model to predict AQI Bucket.

    - **Random Forest Classifier Model**
    - **Evaluation Metrics**
    """)

    data = st.session_state['data']

    # Preparing Data for Modeling
    st.header("Data Preparation for Modeling :dart:")
    features = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene']
    X = data[features]
    y = LabelEncoder().fit_transform(data['AQI_Bucket'])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    st.header("Modeling :gear:")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Classification Report
    st.header("Model Evaluation :memo:")
    st.write(classification_report(y_test, y_pred))

    # Feature Importance
    st.header("Feature Importance :zap:")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    st.write(importance_df)

# -------------------------------------
# Sidebar Navigation
# -------------------------------------
st.sidebar.title("Navigation :compass:")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Preprocessing", "Exploratory Data Analysis (EDA)", "Modeling and Prediction"])

# Page navigation logic
if page == "Data Overview":
    data_overview()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Exploratory Data Analysis (EDA)":
    eda()
elif page == "Modeling and Prediction":
    modeling()
