import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

# Sidebar Header
st.sidebar.title("Machine Learning Dashboard")
st.sidebar.markdown("""
- Upload your dataset
- Clean and explore data
- Train and evaluate machine learning models interactively!
""")

# Title with Animated Banner
st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#4A90E2; font-size: 42px;">
        Interactive Machine Learning Platform 🚀
    </h1>
</div>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Handle new file upload
if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
    st.session_state["original_data"] = pd.read_csv(uploaded_file)
    st.session_state["cleaned_data"] = st.session_state["original_data"].copy()
    st.session_state["uploaded_file"] = uploaded_file
    st.success("New dataset uploaded successfully!")

# Work with the uploaded or reset data
if st.session_state["original_data"] is not None:
    data = st.session_state["cleaned_data"]

    # Dataset Preview
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Reset Dataset Button
    if st.sidebar.button("Reset Dataset"):
        st.session_state["cleaned_data"] = st.session_state["original_data"].copy()
        st.success("Dataset reset to its original state!")
        st.experimental_rerun()

    # Data Cleaning Options
    st.sidebar.header("Data Cleaning Options")

    # Encode Categorical Data
    if st.sidebar.checkbox("Encode Categorical Data"):
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
            st.warning("No categorical columns found to encode.")
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
        st.write(f"Missing values handled using: {method}")
        st.session_state["cleaned_data"] = data

    # Visualize Columns
    st.sidebar.header("Visualization")
    selected_col = st.sidebar.selectbox("Select a column to visualize", data.columns)
    if data[selected_col].dtype in ["float64", "int64"]:
        st.write(f"### Distribution of {selected_col}")
        fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"### Frequency of {selected_col}")
        fig = px.bar(data[selected_col].value_counts().reset_index(), x="index", y=selected_col,
                     labels={"index": selected_col, selected_col: "Count"},
                     title=f"Frequency of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

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
            ["Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]
        )

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

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Display Metrics
        st.write("### Model Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
        st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
        st.write("F1-Score:", f1_score(y_test, y_pred, average="weighted"))

        # Feature Importance for Tree-Based Models
        if model_choice in ["Random Forest", "Decision Tree"]:
            feature_importances = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.write("### Feature Importance")
            fig = px.bar(feature_importances, x="Importance", y="Feature", orientation="h", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("Developed by Ranadeep Mahendra 🚀")
