import logging

# Configure logging
logging.basicConfig(
    filename="app_errors.log", 
    level=logging.ERROR, 
    format="%(asctime)s %(levelname)s %(message)s"
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
import tracemalloc
import gc

# Initialize memory tracking
tracemalloc.start(10)  # Retain 10 frames of traceback for debugging memory leaks

# Initialize session state
if "original_data" not in st.session_state:
    st.session_state["original_data"] = None
if "cleaned_data" not in st.session_state:
    st.session_state["cleaned_data"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# Caching functions
@st.cache_data
def load_dataset(file):
    """Loads the dataset with caching."""
    return pd.read_csv(file)

@st.cache_resource
def process_data(df):
    """Processes the data (e.g., encoding, normalization)."""
    return df.copy()

# Reset app
def reset_app():
    """Resets the app and reloads the page."""
    st.session_state.clear()
    st.experimental_set_query_params()  # Triggers a page reload

# Memory debugging utility
def compare_snapshots():
    """Compares memory snapshots to detect leaks."""
    snapshot = tracemalloc.take_snapshot()
    if "snapshot" in st.session_state:
        diff = snapshot.compare_to(st.session_state["snapshot"], "lineno")
        leaks = [d for d in diff if d.count_diff > 0]
        if leaks:
            st.write("### Potential Memory Leaks Detected")
            for leak in leaks[:10]:  # Limit to top 10 leaks for clarity
                st.write(leak)
    st.session_state["snapshot"] = snapshot

# Title
st.title("Interactive ML Model Testing and EDA Platform 🚀")
st.markdown("""
Upload your dataset to clean, analyze, and test six powerful machine learning algorithms. 
This platform combines data preparation, exploratory data analysis (EDA), and ML model experimentation in a single interface.
""")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset in CSV format", type=["csv"])

try:
    # Handle new file upload
    if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
        try:
            data = load_dataset(uploaded_file)

            # Dataset size check (limit to 100 MB)
            if data.memory_usage(deep=True).sum() > 100 * 1024 * 1024:
                st.error("Dataset is too large! Please upload a file smaller than 100 MB.")
                st.stop()

            st.session_state["original_data"] = data.copy()
            st.session_state["cleaned_data"] = data.copy()
            st.session_state["uploaded_file"] = uploaded_file
            st.success("New dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            logging.error(f"Error uploading file: {e}", exc_info=True)

    # Work with the uploaded data
    if st.session_state["original_data"] is not None:
        try:
            data = st.session_state["cleaned_data"]

            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(data)

            # Option to reset the dataset
            if st.sidebar.button("Reset Dataset"):
                st.session_state["cleaned_data"] = st.session_state["original_data"].copy()
                st.success("Dataset reset to its original state!")
                st.experimental_set_query_params()  # Triggers app reload
        except Exception as e:
            st.error(f"Error working with uploaded data: {e}")
            logging.error(f"Error working with uploaded data: {e}", exc_info=True)

        # Data Cleaning Options
        st.sidebar.header("Data Cleaning Options")

        # Handle Missing Values
        if st.sidebar.checkbox("Handle Missing Values"):
            method = st.sidebar.selectbox("Choose a method", ["Mean", "Median", "Mode", "Drop Rows"])
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

            try:
                if method == "Mean":
                    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                elif method == "Median":
                    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                elif method == "Mode":
                    for col in data.columns:
                        data[col].fillna(data[col].mode()[0], inplace=True)
                elif method == "Drop Rows":
                    data.dropna(inplace=True)
                st.success(f"Missing values handled using: {method}")
                st.session_state["cleaned_data"] = data
            except Exception as e:
                st.error(f"Error handling missing values: {e}")
                logging.error(f"Error handling missing values: {e}", exc_info=True)

        # Automatically Encode Categorical Data
        if st.sidebar.checkbox("Encode Categorical Data"):
            try:
                encoding_method = st.sidebar.radio("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
                categorical_cols = data.select_dtypes(include=["object"]).columns

                if len(categorical_cols) > 0:
                    if encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        for col in categorical_cols:
                            data[col] = le.fit_transform(data[col].astype(str))
                        st.success("Label Encoding applied.")
                    elif encoding_method == "One-Hot Encoding":
                        data = pd.get_dummies(data, columns=categorical_cols)
                        st.success("One-Hot Encoding applied.")
                else:
                    st.info("No categorical columns found for encoding.")
                st.session_state["cleaned_data"] = data
            except Exception as e:
                st.error(f"Error encoding categorical data: {e}")
                logging.error(f"Error encoding categorical data: {e}", exc_info=True)

        # Display cleaned dataset
        st.write("### Cleaned Dataset Preview")
        st.dataframe(data)

        # EDA Options
        st.sidebar.header("EDA Options")

        if st.sidebar.checkbox("Show Summary Statistics"):
            try:
                st.write("### Summary Statistics")
                st.write(data.describe(include="all"))
            except Exception as e:
                st.error(f"Error generating summary statistics: {e}")
                logging.error(f"Error generating summary statistics: {e}", exc_info=True)

        if st.sidebar.checkbox("Correlation Heatmap"):
            try:
                numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
                if len(numeric_cols) > 0:
                    st.write("### Correlation Heatmap")
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
                    st.pyplot(plt)
                else:
                    st.info("No numeric columns for heatmap.")
            except Exception as e:
                st.error(f"Error generating correlation heatmap: {e}")
                logging.error(f"Error generating correlation heatmap: {e}", exc_info=True)

        if st.sidebar.checkbox("Visualize Columns"):
            try:
                selected_col = st.sidebar.selectbox("Select a column to visualize", data.columns)
                if data[selected_col].dtype in ["float64", "int64"]:
                    st.write(f"### Distribution of {selected_col}")
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data[selected_col], kde=True)
                    st.pyplot(plt)
                else:
                    st.write(f"### Frequency of {selected_col}")
                    plt.figure(figsize=(8, 4))
                    sns.countplot(y=data[selected_col])
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error visualizing column: {e}")
                logging.error(f"Error visualizing column: {e}", exc_info=True)

        # Machine Learning Section
        st.sidebar.header("Machine Learning")
        if st.sidebar.checkbox("Run ML Models"):
            try:
                target_column = st.sidebar.selectbox("Select Target Column", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Encode categorical variables
                X = pd.get_dummies(X)
                if y.dtype == "object":
                    y = LabelEncoder().fit_transform(y)

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Model Selection
                model_choice = st.sidebar.radio(
                    "Choose ML Model",
                    ["Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "K-Means Clustering"]
                )

                if model_choice == "K-Means Clustering":
                    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                    model = KMeans(n_clusters=num_clusters, random_state=42)
                    model.fit(X)
                    st.write("### Clustering Results")
                    st.write("Cluster Assignments:", model.labels_)
                    st.write("Inertia (Sum of squared distances):", model.inertia_)
                    if X.shape[1] > 1:
                        silhouette_avg = silhouette_score(X, model.labels_)
                        st.write("Silhouette Score:", silhouette_avg)
                else:
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=50)
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeClassifier()
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression()
                    elif model_choice == "K-Nearest Neighbors":
                        model = KNeighborsClassifier()
                    elif model_choice == "Support Vector Machine":
                        model = SVC(probability=True)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.write("### Model Performance")
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))
                    st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
                    st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
                    st.write("F1-Score:", f1_score(y_test, y_pred, average="weighted"))

                    # Confusion Matrix
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error training model: {e}")
                logging.error(f"Error training model: {e}", exc_info=True)
        gc.collect()
        compare_snapshots()

except Exception as e:
    st.error(f"Unexpected error occurred: {e}")
    logging.error(f"Unexpected error occurred: {e}", exc_info=True)
    if st.sidebar.button("Reset App"):
        reset_app()

# Footer
st.sidebar.markdown("Developed by Ranadeep Mahendra 🚀")
