import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Encoder for the target variable
def binary_encoder(value):
    binary_encoder = {"B": 1, "M": 0}
    return binary_encoder.get(value)

# Streamlit App
st.title("Breast Cancer Classification App")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Upload Data", "Data Exploration", "Train Model", "Predict"])

if page == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Drop unnecessary columns
        data = data.drop(columns=["id", "Unnamed: 32"], errors='ignore')
        st.session_state['data'] = data
        st.success("Dataset uploaded successfully!")

elif page == "Data Exploration":
    st.header("Data Exploration")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Dataset Statistics")
        st.write(data.describe())

        st.subheader("Class Distribution")
        if 'diagnosis' in data.columns:
            st.bar_chart(data['diagnosis'].value_counts())
        else:
            st.error("The dataset must contain a 'diagnosis' column.")
    else:
        st.error("Please upload a dataset first.")

elif page == "Train Model":
    st.header("Train Logistic Regression Model")
    if 'data' in st.session_state:
        data = st.session_state['data']

        if 'diagnosis' in data.columns:
            X = data.drop(columns=['diagnosis'])
            y = data['diagnosis'].map(binary_encoder)

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

            st.text("Accuracy:")
            st.write(accuracy_score(y_test, y_pred))

            st.session_state['model'] = model
            st.session_state['imputer'] = imputer
            st.session_state['columns'] = data.drop(columns=['diagnosis']).columns.tolist()
        else:
            st.error("The dataset must contain a 'diagnosis' column.")
    else:
        st.error("Please upload a dataset first.")

elif page == "Predict":
    st.header("Make Predictions")
    if 'model' in st.session_state and 'imputer' in st.session_state and 'columns' in st.session_state:
        model = st.session_state['model']
        imputer = st.session_state['imputer']
        columns = st.session_state['columns']

        sample_input = st.text_area("Enter input features (comma-separated, matching column order):")
        if sample_input:
            try:
                input_data = [float(i) for i in sample_input.split(',')]
                if len(input_data) != len(columns):
                    st.error(f"Expected {len(columns)} values, but got {len(input_data)}.")
                else:
                    input_df = pd.DataFrame([input_data], columns=columns)
                    input_df = imputer.transform(input_df)

                    prediction = model.predict(input_df)
                    result = "Benign" if prediction[0] == 1 else "Malignant"
                    st.write(f"Prediction: {result}")
            except ValueError:
                st.error("Please enter valid numeric values, separated by commas.")
    else:
        st.error("Please train a model first.")
