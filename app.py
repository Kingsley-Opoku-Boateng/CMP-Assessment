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
    st.title("Data Overview ")

   
  
    data = st.session_state['data']

    # Dataset Information
    st.header("Dataset Information")

    st.markdown("""

    The Beijing Air Quality dataset contains hourly readings of pollutants  PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3 
    and Volatile Organic Compounds (VOCs): Benzene, Toluene, Xylene,  Air Quality Index (AQI) and AQI_Bucket  across 26 cities.

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
    pollutants = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene']
    for pollutant in pollutants:
        st.subheader(f"{pollutant} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[pollutant], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation Matrix
    st.header("Correlation Matrix")
    corr_matrix = data[['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene']].corr()
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

    # Preparing Data for Modeling
    st.header("Data Preparation for Modeling")
    features = ['PM2.5', 'PM10', 'NO','NO2', 'NOx', 'NH3','CO','SO2','O3','Benzene', 'Toluene', 'Xylene']
    X = data[features]
    y = LabelEncoder().fit_transform(data['AQI_Bucket'])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    st.header("Modeling")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

  def modeling():
    st.title("Predict Air Quality (AQI)")

    # Load the pre-trained model (you can replace this with a saved model if necessary)
    model = load_trained_model()

    # Pollutant Features for input
    st.header("Enter Pollutant Values to Predict AQI")

    # Create input fields for the user to input values
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=0.1)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=0.1)
    no = st.number_input("NO (µg/m³)", min_value=0.0, step=0.1)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1)
    nox = st.number_input("NOx (µg/m³)", min_value=0.0, step=0.1)
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, step=0.1)
    co = st.number_input("CO (µg/m³)", min_value=0.0, step=0.1)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1)
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, step=0.1)
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, step=0.1)
    toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, step=0.1)
    xylene = st.number_input("Xylene (µg/m³)", min_value=0.0, step=0.1)

    # Combine all inputs into a list (feature vector)
    features = [pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]

    # Label encoder for AQI buckets (assuming you've trained your model this way)
    label_encoder = LabelEncoder()
    label_encoder.fit(['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])

    # Prediction Button
    if st.button("Predict AQI"):
        # Ensure we have valid input
        if all(val >= 0 for val in features):
            # Predict AQI based on input features
            prediction = model.predict([features])

            # Decode the prediction back to the AQI bucket
            predicted_aqi = label_encoder.inverse_transform(prediction)

            # Display result
            st.write(f"The predicted AQI category is: **{predicted_aqi[0]}**")
        else:
            st.error("Please enter valid positive values for all pollutants.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Modeling and Prediction"])

if page == "Modeling and Prediction":
    modeling()


