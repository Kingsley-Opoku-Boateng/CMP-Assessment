import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
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
    st.title("Data Overview")

    data = st.session_state['data']

    # Dataset Information
    st.header("Dataset Information")
    st.markdown("""
    The Beijing Air Quality dataset contains hourly readings of pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, 
    and Volatile Organic Compounds (VOCs): Benzene, Toluene, Xylene, Air Quality Index (AQI) and AQI_Bucket across 26 cities.
    """)

    # Preview data
    st.header("Data Preview")
    num_rows = st.slider("Select number of rows to preview:", 1, 100, 10)
    st.write(data.head(num_rows))

    # Descriptive Statistics
    st.header("Descriptive Statistics")
    show_desc_table = st.checkbox("Show Descriptive Statistics")
    if show_desc_table:
        st.write(data.describe())

    # Missing values
    st.header("Missing Values Analysis")
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
    st.write(missing_df)

# -------------------------------------
# Page 2: Data Preprocessing
# -------------------------------------
def data_preprocessing():
    st.title("Data Preprocessing")

    data = st.session_state['data']

    # Handle Missing Data
    st.header("Handling Missing Data")
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
    st.header("Feature Engineering")
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
    st.header("Dropping Columns")
    columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
    if st.button("Drop Columns"):
        data.drop(columns=columns_to_drop, inplace=True)
        st.success("Columns dropped successfully.")
        st.session_state['data'] = data

# -------------------------------------
# Page 3: Exploratory Data Analysis (EDA)
# -------------------------------------
def eda():
    st.title("Exploratory Data Analysis (EDA)")

    data = st.session_state['data']

    # Pollutant Distribution
    st.header("Pollutant Distribution")
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    for pollutant in pollutants:
        st.subheader(f"{pollutant} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[pollutant], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation Matrix
    st.header("Correlation Matrix")
    corr_matrix = data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].corr()
    st.write(corr_matrix)

    # Heatmap of Correlation
    st.header("Correlation Heatmap")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)


# -------------------------------------
# Page 4: Modeling and Prediction
# -------------------------------------
def modeling():
    st.title("Modeling and Prediction")

    data = st.session_state['data']

    # -------------------------------
    # Prepare data
    # -------------------------------
    st.header("Model Training")

    features = [
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
        'NH3', 'CO', 'SO2', 'O3',
        'Benzene', 'Toluene', 'Xylene'
    ]

    # Drop rows with missing values (simple & fast)
    model_data = data[features + ['AQI_Bucket']].dropna()

    X = model_data[features]
    y = model_data['AQI_Bucket']

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.header("Predict AQI Category")

    st.write("Enter pollutant and VOC values below:")

    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0)
    no = st.number_input("NO (µg/m³)", 0.0)
    no2 = st.number_input("NO2 (µg/m³)", 0.0)
    nox = st.number_input("NOx (µg/m³)", 0.0)
    nh3 = st.number_input("NH3 (µg/m³)", 0.0)
    co = st.number_input("CO (µg/m³)", 0.0)
    so2 = st.number_input("SO2 (µg/m³)", 0.0)
    o3 = st.number_input("O3 (µg/m³)", 0.0)
    benzene = st.number_input("Benzene (µg/m³)", 0.0)
    toluene = st.number_input("Toluene (µg/m³)", 0.0)
    xylene = st.number_input("Xylene (µg/m³)", 0.0)

    input_data = np.array([[
        pm25, pm10, no, no2, nox,
        nh3, co, so2, o3,
        benzene, toluene, xylene
    ]])

    if st.button("Predict AQI"):
        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)

        st.success(f"Predicted AQI Category: **{predicted_label[0]}**")


# -------------------------------------
# Sidebar Navigation
# -------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Data Overview",
        "Data Preprocessing",
        "Exploratory Data Analysis (EDA)",
        "Modeling and Prediction"
    ]
)

if page == "Data Overview":
    data_overview()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Exploratory Data Analysis (EDA)":
    eda()
elif page == "Modeling and Prediction":
    modeling()


