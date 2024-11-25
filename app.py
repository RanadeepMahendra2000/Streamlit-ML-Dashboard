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
    confusion_matrix, silhouette_score
)

# Sidebar Header
st.sidebar.title("Machine Learning Dashboard")
st.sidebar.markdown("""
- **Step 1**: Upload your dataset (CSV format).
- **Step 2**: Clean and explore your data.
- **Step 3**: Select and train machine learning models.
- **Step 4**: Evaluate performance and understand results.
""")

# Title with Background
st.title("Interactive Machine Learning Platform 🚀")
st.markdown("""
This platform guides you through a complete **Machine Learning workflow**, including:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Model Training and Evaluation

📚 **Learn about the models**: Hover over each model name in the sidebar to see what it does!
""")

# Machine Learning Background Section
st.sidebar.subheader("About the Models")
st.sidebar.markdown("""
- **Random Forest**: A collection of decision trees that vote to improve accuracy.
- **Decision Tree**: A tree structure that splits data based on features for predictions.
- **Logistic Regression**: A statistical model for binary classification.
- **K-Nearest Neighbors (KNN)**: A simple algorithm that predicts based on the closest neighbors.
- **Support Vector Machine (SVM)**: Finds a hyperplane to separate data into classes.
- **K-Means Clustering**: Groups data points into clusters based on feature similarity.
""")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Handle uploaded file
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Step-by-Step Workflow
    st.header("Machine Learning Workflow")
    st.subheader("1. Data Cleaning")
    st.markdown("""
    - Handle missing values, duplicates, or outliers.
    - Encode categorical variables into numerical formats.
    """)
    
    # Handle Missing Values
    if st.sidebar.checkbox("Handle Missing Values"):
        method = st.sidebar.radio("Choose a method to handle missing values", ["Mean", "Median", "Mode", "Drop Rows"])
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
        st.success(f"Missing values handled using: {method}")

    # Categorical Encoding
    if st.sidebar.checkbox("Encode Categorical Data"):
        encoding_method = st.sidebar.radio("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
        categorical_cols = data.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in categorical_cols:
                    data[col] = le.fit_transform(data[col].astype(str))
                st.success("Applied Label Encoding.")
            elif encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=categorical_cols)
                st.success("Applied One-Hot Encoding.")

    st.subheader("2. Exploratory Data Analysis (EDA)")
    st.markdown("""
    - Understand your data visually before training models.
    """)
    
    # Correlation Heatmap
    if st.sidebar.checkbox("Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        corr = data.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # Machine Learning Section
    st.subheader("3. Model Training and Evaluation")
    target_column = st.selectbox("Select the Target Column (for supervised models)", options=[None] + list(data.columns))
    X = data.drop(columns=[target_column]) if target_column else data
    y = data[target_column] if target_column else None

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    # Train-Test Split for supervised models
    if target_column:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Selection
    model_choice = st.sidebar.radio(
        "Choose a Machine Learning Model",
        ["Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "K-Means Clustering"]
    )

    if st.sidebar.button("Train Model"):
        if model_choice == "K-Means Clustering":
            # K-Means Clustering
            num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=num_clusters, random_state=42)
            model.fit(X)
            data["Cluster"] = model.labels_

            st.write("### Clustering Results")
            st.write("Cluster Assignments:", model.labels_)
            st.write("Inertia (Sum of squared distances to cluster centers):", model.inertia_)

            # Silhouette Score
            if len(X) >= num_clusters:
                silhouette_avg = silhouette_score(X, model.labels_)
                st.write("Silhouette Score:", silhouette_avg)

            # Visualize Clusters (only works if there are two features or PCA-reduced)
            if X.shape[1] == 2:
                fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color=data["Cluster"].astype(str),
                                 title="K-Means Clustering Visualization", labels={"color": "Cluster"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Clustering visualization requires exactly 2 numeric features. Consider reducing dimensions.")

        else:
            # Supervised Models
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

            # Train supervised model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display Metrics
            st.write(f"### {model_choice} Performance")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
            st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
            st.write("F1-Score:", f1_score(y_test, y_pred, average="weighted"))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels={"x": "Predicted", "y": "Actual"})
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("Developed by Ranadeep Mahendra 🚀")
