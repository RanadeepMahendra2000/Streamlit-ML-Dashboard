import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, silhouette_score
)

# Initialize session state
if "original_data" not in st.session_state:
    st.session_state["original_data"] = None
if "cleaned_data" not in st.session_state:
    st.session_state["cleaned_data"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# Title
st.title("Interactive ML Model Testing and EDA Platform")
st.markdown("""
Upload your dataset to clean, analyze, and test six powerful machine learning algorithms. 
This platform combines data preparation, exploratory data analysis (EDA), and ML model experimentation in a single interface.
""")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset in CSV format", type=["csv"])

# Handle new file upload
if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
    st.session_state["original_data"] = pd.read_csv(uploaded_file)
    st.session_state["cleaned_data"] = st.session_state["original_data"].copy()
    st.session_state["uploaded_file"] = uploaded_file
    st.success("New dataset uploaded successfully!")

# Work with the uploaded or reset data
if st.session_state["original_data"] is not None:
    data = st.session_state["cleaned_data"]

    # Display dataset
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Option to reset the dataset
    if st.sidebar.button("Reset Dataset"):
        st.session_state["cleaned_data"] = st.session_state["original_data"].copy()
        st.success("Dataset reset to its original state!")
        st.experimental_rerun()

    # Data Cleaning Options
    st.sidebar.header("Data Cleaning Options")

    # Automatically Encode Categorical Data
    if st.sidebar.checkbox("Automatically Encode Categorical Data"):
        encoding_method = st.sidebar.radio("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) > 0:
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in categorical_cols:
                    data[col] = le.fit_transform(data[col].astype(str))
                st.write("Applied Label Encoding to categorical columns.")
            elif encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=categorical_cols)
                st.write("Applied One-Hot Encoding to categorical columns.")
        else:
            st.write("No categorical columns found to encode.")
        st.session_state["cleaned_data"] = data

    # Handle Missing Values
    if st.sidebar.checkbox("Handle Missing Values"):
        method = st.sidebar.selectbox("Choose a method", ["Mean", "Median", "Mode", "Drop Rows"])
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

        if method == "Mean":
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        elif method == "Median":
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        elif method == "Mode":
            for col in data.columns:
                data[col].fillna(data[col].mode()[0], inplace=True)
        elif method == "Drop Rows":
            data.dropna(inplace=True)
        st.write("Missing values handled using:", method)
        st.session_state["cleaned_data"] = data

    # Drop Duplicates
    if st.sidebar.checkbox("Drop Duplicates"):
        data.drop_duplicates(inplace=True)
        st.write("Duplicates dropped.")
        st.session_state["cleaned_data"] = data

    # Normalize/Standardize Data
    if st.sidebar.checkbox("Normalize/Standardize Data"):
        norm_method = st.sidebar.selectbox("Choose a method", ["Standardization", "Min-Max Scaling"])
        scaler = StandardScaler() if norm_method == "Standardization" else MinMaxScaler()
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        st.write(f"Data {norm_method} applied to numeric columns.")
        st.session_state["cleaned_data"] = data

    st.write("### Cleaned Dataset Preview")
    st.dataframe(data)

    # EDA Options
    st.sidebar.header("EDA Options")

    if st.sidebar.checkbox("Show Summary Statistics"):
        st.write("### Summary Statistics")
        st.write(data.describe(include="all"))

    if st.sidebar.checkbox("Correlation Heatmap"):
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) > 0:
            st.write("### Correlation Heatmap")
            plt.figure(figsize=(10, 6))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.write("No numeric columns available for correlation heatmap.")

    if st.sidebar.checkbox("Visualize Columns"):
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

    # Machine Learning Section
    st.sidebar.header("Machine Learning")
    if st.sidebar.checkbox("Run Machine Learning Models"):
        target_column = st.sidebar.selectbox("Select Target Column", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols)

        # Encode target column if categorical
        if y.dtype in ["object", "category"]:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train-Test Split
        test_size = st.sidebar.slider("Test Size (Proportion)", 0.1, 0.5, 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model Selection
        model_choice = st.sidebar.selectbox(
            "Choose an ML Model",
            ["Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "K-Means Clustering"]
        )

        if model_choice == "K-Means Clustering":
            model = KMeans(n_clusters=3)
            model.fit(X)
            st.write("### Clustering Results")
            st.write("Cluster Assignments:", model.labels_)
            st.write("Inertia (Sum of squared distances to cluster centers):", model.inertia_)
            if len(X) >= 2:
                silhouette_avg = silhouette_score(X, model.labels_)
                st.write("Silhouette Score:", silhouette_avg)
        else:
            # Initialize and train supervised model
            if model_choice == "Random Forest":
                model = RandomForestClassifier()
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

            # Display metrics
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

# Footer
st.sidebar.markdown("Developed by Ranadeep Mahendra")
