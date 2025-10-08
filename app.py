import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

st.set_page_config(page_title="Smart ML App", layout="wide")

st.title("Smart ML Analyzer & Model Builder")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Raw Dataset Preview")
    st.dataframe(df.head())

    # Clean null values
    st.subheader("🧹 Data Cleaning")
    st.write(f"Initial shape: {df.shape}")
    df = df.dropna()
    st.write(f"After removing nulls: {df.shape}")

    # Show basic stats
    st.subheader("Dataset Summary")
    st.write(df.describe())

    # EDA
    st.subheader("Exploratory Data Analysis")

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pairplot
    if len(df.columns) <= 8:
        st.write("### Pairplot (for small datasets)")
        fig2 = sns.pairplot(df)
        st.pyplot(fig2)

    # Model selection
    st.subheader("Machine Learning Model Selection")
    model_type = st.selectbox(
        "Choose a Machine Learning model type:",
        ["Regression", "Classification"]
    )

    # Select target column
    target_col = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "Regression":
        model_name = st.selectbox(
            "Select Regression Model:",
            ["Linear Regression", "Random Forest Regressor"]
        )

        if model_name == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance (Regression)")
        st.write("**R² Score:**", r2_score(y_test, y_pred))
        st.write("**Mean Squared Error:**", mean_squared_error(y_test, y_pred))

        st.subheader(" Predicted vs Actual")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred, color='blue')
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        ax3.set_title("Actual vs Predicted")
        st.pyplot(fig3)

    else:
        model_name = st.selectbox(
            "Select Classification Model:",
            ["Logistic Regression", "Random Forest Classifier"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(" Model Performance (Classification)")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
        st.text("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        st.write("### Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax4)
        st.pyplot(fig4)

    st.success(" Model training and analysis complete!")

else:
    st.info("Please upload a CSV file to get started.")
