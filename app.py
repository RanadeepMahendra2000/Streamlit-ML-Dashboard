

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
        if method == "Mean":
            data.fillna(data.mean(), inplace=True)
        elif method == "Median":
            data.fillna(data.median(), inplace=True)
        elif method == "Mode":
            for col in data.select_dtypes(include=["object", "category"]).columns:
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

# Footer
st.sidebar.markdown("Ranadeep Mahendra")
