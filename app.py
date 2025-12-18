import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Caching the data loading function for faster access
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/Kingsley-Opoku-Boateng/CMP-Assessment/main/air_quality_data.csv"
    df = pd.read_csv(url)
    return df

# Main function for the app
def main():
    st.title('Air Quality Prediction App')

    # Load Data
    df = load_data()

    # Data Overview Section
    st.header('Data Overview')
    st.write("Dataset Information:")
    st.write(df.info())  # Display dataset info
    st.write("First few rows of the dataset:")
    st.write(df.head())  # Show the first few rows of the data

    # Exploratory Data Analysis (EDA)
    st.header('Exploratory Data Analysis (EDA)')
    
    # Average values of pollutants and VOCs
    st.subheader('Average Pollutant and VOC Concentrations')
    avg_pollutants = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Benzene', 'Toluene', 'Xylene']].mean()
    st.write(avg_pollutants)

    # Display simple distributions of pollutants
    st.subheader('Pollutant and VOC Distributions')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Benzene', 'Toluene', 'Xylene']

    for i, pollutant in enumerate(pollutants):
        sns.histplot(df[pollutant], ax=axes[i], kde=True)
        axes[i].set_title(f'{pollutant} Distribution')

    st.pyplot(fig)

    # Correlation Matrix Section
    st.header('Correlation Between Pollutants and VOCs')
    correlation_matrix = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Benzene', 'Toluene', 'Xylene']].corr()
    
    # Plotting the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    st.pyplot()

    # Modeling and Prediction Section
    st.header('Modeling and Prediction')

    # Preprocessing the data for the model
    df['AQI_Bucket'] = df['AQI_Bucket'].apply(LabelEncoder().fit_transform)  # Convert AQI_Bucket to numerical values

    # Selecting features and target variable
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Benzene', 'Toluene', 'Xylene']
    X = df[features]
    y = df['AQI_Bucket']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model: Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Displaying model performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy: ", accuracy)
    st.write("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

   

    

