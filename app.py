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
    confusion_matrix, roc_auc_score, roc_curve, silhouette_score
)

# Title
st.title("Interactive ML Model Testing and EDA Platform")
st.markdown("""
Upload your dataset to clean, analyze, and test six powerful machine learning algorithms. 
This platform combines data preparation, exploratory data analysis (EDA), and ML model experimentation in a single interface.
""")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset in CSV format", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Data Cleaning Options
    st.sidebar.header("Data Cleaning Options")

    # Handle Missing Values
    if st.sidebar.checkbox("Handle Missing Values"):
        method = st.sidebar.selectbox("Choose a method", ["Mean", "Median", "Mode", "Drop Rows"])
    
    # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        non_numeric_cols = data.select_dtypes(exclude=["float64", "int64"]).columns
    
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

    # Handle Outliers
    if st.sidebar.checkbox("Handle Outliers"):
        outlier_method = st.sidebar.selectbox("Outlier Detection Method", ["Z-Score", "IQR"])
        if outlier_method == "Z-Score":
            from scipy.stats import zscore
            z_scores = zscore(data.select_dtypes(include=["float64", "int64"]))
            data = data[(np.abs(z_scores) < 3).all(axis=1)]
        elif outlier_method == "IQR":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.write("Outliers handled using:", outlier_method)

    # Drop Duplicates
    if st.sidebar.checkbox("Drop Duplicates"):
        data.drop_duplicates(inplace=True)
        st.write("Duplicates dropped.")

    # Standardize/Normalize Data
    if st.sidebar.checkbox("Normalize/Standardize Data"):
        norm_method = st.sidebar.selectbox("Choose a method", ["Standardization", "Min-Max Scaling"])
        scaler = StandardScaler() if norm_method == "Standardization" else MinMaxScaler()
        num_cols = data.select_dtypes(include=["float64", "int64"]).columns
        data[num_cols] = scaler.fit_transform(data[num_cols])
        st.write(f"Data {norm_method} applied to numeric columns.")

    # Encoding Categorical Variables
    if st.sidebar.checkbox("Encode Categorical Variables"):
        encoding_method = st.sidebar.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in data.select_dtypes(include=["object", "category"]).columns:
                data[col] = le.fit_transform(data[col])
        elif encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data)
        st.write("Categorical variables encoded using:", encoding_method)

    st.write("### Cleaned Dataset Preview")
    st.dataframe(data)

    # EDA Options
    st.sidebar.header("EDA Options")

    if st.sidebar.checkbox("Show Summary Statistics"):
        st.write("### Summary Statistics")
        st.write(data.describe())

    if st.sidebar.checkbox("Visualize Missing Values"):
        st.write("### Missing Value Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        st.pyplot(plt)

    if st.sidebar.checkbox("Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    if st.sidebar.checkbox("Distribution Visualizations"):
        st.write("### Distribution Plots")
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            plt.figure()
            sns.histplot(data[col], kde=True)
            st.pyplot(plt)

    if st.sidebar.checkbox("Box Plots"):
        st.write("### Box Plots for Outlier Detection")
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            plt.figure()
            sns.boxplot(data[col])
            st.pyplot(plt)

    if st.sidebar.checkbox("Pair Plots"):
        st.write("### Pair Plot of Variables")
        sns.pairplot(data)
        st.pyplot(plt)

    # Machine Learning Section
    st.sidebar.header("Machine Learning")
    if st.sidebar.checkbox("Run Machine Learning Models"):
        # Model Selection
        model_choice = st.sidebar.selectbox(
            "Choose an ML Model",
            [
                "Random Forest",
                "Decision Tree",
                "Logistic Regression",
                "K-Nearest Neighbors (KNN)",
                "Support Vector Machine (SVM)",
                "K-Means Clustering",
            ],
        )

        # Target Variable Selection
        target_column = st.sidebar.selectbox("Select Target Column", data.columns)

        # Feature and Target Data
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Train-Test Split
        test_size = st.sidebar.slider("Test Size (Proportion)", 0.1, 0.5, 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model Initialization
        try:
            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier()
            elif model_choice == "Support Vector Machine (SVM)":
                model = SVC(probability=True)
            elif model_choice == "K-Means Clustering":
                model = KMeans(n_clusters=3)

            # Fit Model and Display Results
            if model_choice == "K-Means Clustering":
                model.fit(X)
                st.write("### Clustering Results")
                st.write("Cluster Assignments:", model.labels_)
                st.write("Inertia (Sum of squared distances to cluster centers):", model.inertia_)
                if len(X) >= 2:
                    silhouette_avg = silhouette_score(X, model.labels_)
                    st.write("Silhouette Score:", silhouette_avg)
            else:
                # Fit classification model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                # Accuracy and Metrics
                st.write("### Model Evaluation")
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

                # ROC Curve
                if y_proba is not None:
                    st.write("### ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    plt.figure(figsize=(10, 6))
                    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba)))
                    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver Operating Characteristic (ROC) Curve")
                    plt.legend()
                    st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.sidebar.markdown("Developed by [Your Name]")
