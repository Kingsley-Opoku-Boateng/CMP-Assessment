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
@st.cache_data
def load_data():
    # Your data loading code here
    url = "https://raw.githubusercontent.com/Kingsley-Opoku-Boateng/CMP-Assessment/main/air_quality_data.csv"
    data = pd.read_csv(url)
    return data


# Sidebar for Navigation
st.sidebar.title("Air Quality Analysis App")
page = st.sidebar.radio("Select a Page", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modeling and Prediction"])

# -------------------- Data Overview --------------------
if page == "Data Overview":
    st.title("Data Overview")
    st.write("This is the overview of the air quality dataset.")
    
    # Display basic dataset info
    st.write("### Dataset Information:")
    st.write(df.info())
    
    # Show a preview of the dataset
    st.write("### Dataset Preview:")
    st.write(df.head())
    
    # Show the count of null values
    st.write("### Null Values:")
    st.write(df.isnull().sum())

# -------------------- Exploratory Data Analysis (EDA) --------------------
if page == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Let's explore the dataset with some visualizations.")
    
    # Correlation heatmap (Including VOCs)
    st.write("### Correlation Heatmap")
    correlation_matrix = df.corr()
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig)
    
    # Scatter Plot of PM2.5 vs PM10
    st.write("### PM2.5 vs PM10 Scatter Plot")
    fig = px.scatter(df, x="PM2.5", y="PM10", color="AQI_Bucket", title="PM2.5 vs PM10")
    st.plotly_chart(fig)
    
    # AQI Distribution
    st.write("### AQI Distribution")
    fig = px.histogram(df, x="AQI", nbins=50, title="Distribution of AQI")
    st.plotly_chart(fig)

    # Box plot for pollutants including VOCs
    st.write("### Box Plot for Pollutants (Including VOCs)")
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, pollutant in enumerate(pollutants):
        axes[i].boxplot(df[pollutant].dropna(), vert=False, patch_artist=True, showfliers=False)
        axes[i].set_title(f'Box Plot of {pollutant}')
        axes[i].set_xlabel(pollutant)
    
    st.pyplot(fig)

    # Pair Plot for VOCs and Pollutants
    st.write("### Pair Plot for VOCs and Pollutants")
    pair_plot_df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'NH3', 'Benzene', 'Toluene', 'Xylene']]
    pair_plot = sns.pairplot(pair_plot_df)
    st.pyplot(pair_plot)

# -------------------- Modeling and Prediction --------------------
if page == "Modeling and Prediction":
    st.title("Modeling and Prediction")
    st.write("Train a model to predict AQI or AQI bucket.")

    # Prepare the data
    df = df.dropna(subset=["AQI_Bucket"])  # Dropping rows with missing target (AQI_Bucket)
    
    # Select features and target (Including VOCs)
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'NH3', 'Benzene', 'Toluene', 'Xylene']  # Numerical pollutants
    categorical_features = ['City', 'Season']  # Categorical variables (Season, City)
    target = 'AQI_Bucket'
    
    X = df[features + categorical_features]
    y = df[target]

    # Preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the RandomForest Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    rf_pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write("### Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Classification Report:\n{class_report}")
    
    # Display confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
    st.pyplot(fig)

    # Model Hyperparameter Tuning (Optional)
    st.write("### Hyperparameter Tuning using GridSearchCV")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
    }
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    st.write(f"Best Parameters: {grid_search.best_params_}")
    st.write(f"Best Score: {grid_search.best_score_}")

    # Download model as pickle (Optional)
    import pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_pipeline, f)
    
    st.write("### Download Model")
    st.download_button(label="Download the Model", data=open('model.pkl', 'rb'), file_name="air_quality_model.pkl", mime="application/octet-stream")

# -------------------- Additional Features --------------------
if page == "Additional Features":
    st.title("Make Predictions")
    st.write("Provide input data to predict AQI category (Good, Moderate, Unhealthy).")

    # Create user input fields
    PM2_5 = st.slider("PM2.5", min_value=0, max_value=500, step=1)
    PM10 = st.slider("PM10", min_value=0, max_value=500, step=1)
    NO2 = st.slider("NO2", min_value=0, max_value=200, step=1)
    CO = st.slider("CO", min_value=0, max_value=2000, step=1)
    O3 = st.slider("O3", min_value=0, max_value=300, step=1)
    SO2 = st.slider("SO2", min_value=0, max_value=500, step=1)
    NH3 = st.slider("NH3", min_value=0, max_value=500, step=1)
    Benzene = st.slider("Benzene", min_value=0, max_value=100, step=1)
    Toluene = st.slider("Toluene", min_value=0, max_value=100, step=1)
    Xylene = st.slider("Xylene", min_value=0, max_value=100, step=1)
    
    # Create a data frame for the input data
    user_input = pd.DataFrame({
        'PM2.5': [PM2_5],
        'PM10': [PM10],
        'NO2': [NO2],
        'CO': [CO],
        'O3': [O3],
        'SO2': [SO2],
        'NH3': [NH3],
        'Benzene': [Benzene],
        'Toluene': [Toluene],
        'Xylene': [Xylene],
        'City': ['Delhi'],  # Static, could be made dynamic
        'Season': ['Summer']  # Static, could be made dynamic
    })

    # Predict the AQI category
    prediction = rf_pipeline.predict(user_input)
    st.write(f"Predicted AQI Category: {prediction[0]}")


   

    
 
