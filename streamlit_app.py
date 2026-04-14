import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Disease Diagnosis Explorer", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("disease_diagnosis.csv")
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    if "Blood_Pressure_mmHg" in df.columns:
        bp = df["Blood_Pressure_mmHg"].astype(str).str.split("/", expand=True)
        bp.columns = ["Systolic_BP", "Diastolic_BP"]
        df = pd.concat([df, bp], axis=1)
        df["Systolic_BP"] = pd.to_numeric(df["Systolic_BP"], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(df["Diastolic_BP"], errors="coerce")
        df = df.drop(columns=["Blood_Pressure_mmHg"])

    return df

@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df2 = df.copy()
    encoders = {}
    categorical_cols = [
        "Gender",
        "Treatment_Plan",
        "Diagnosis",
        "Symptom_1",
        "Symptom_2",
        "Symptom_3",
        "Severity",
    ]

    for col in categorical_cols:
        if col in df2.columns:
            encoder = LabelEncoder()
            df2[col] = encoder.fit_transform(df2[col].astype(str))
            encoders[col] = encoder

    if "Severity" in df2.columns:
        df2["Severity_Encoded"] = encoders["Severity"].transform(
            df["Severity"].astype(str)
        )

    return df2, encoders

@st.cache_resource
def train_model(df2: pd.DataFrame):
    target = "Severity_Encoded"
    drop_cols = ["Severity", "Severity_Encoded", "Diagnosis", "Treatment_Plan"]
    features = [c for c in df2.columns if c not in drop_cols]

    X = df2[features]
    y = df2[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return model, features, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred

def main():
    st.title("Disease Diagnosis Analysis")
    st.markdown(
        "This Streamlit app explores the `disease_diagnosis.csv` dataset and trains a baseline classification model for disease severity."
    )

    df = load_data()
    df2, encoders = preprocess_data(df)
    model, features, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred = (
        train_model(df2)
    )

    page = st.sidebar.selectbox(
        "Navigation",
        ["Overview", "EDA", "Model Training", "Prediction"],
    )

    if page == "Overview":
        st.header("Project Overview")
        st.markdown(
            "This app loads the disease diagnosis dataset, performs exploratory data analysis, preprocesses categorical features, and trains a Decision Tree classifier for severity prediction."
        )
        st.subheader("Dataset sample")
        st.dataframe(df.head(10))
        st.subheader("Column details")
        st.write(df.dtypes)
        st.subheader("Dataset dimensions")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First records")
            st.dataframe(df.head())
            st.subheader("Summary statistics")
            st.dataframe(df.describe(include="all"))
        with col2:
            st.subheader("Missing values")
            st.dataframe(df.isnull().sum().to_frame("missing_count"))
            if "Severity" in df.columns:
                st.subheader("Severity distribution")
                st.bar_chart(df["Severity"].value_counts())

        st.subheader("Categorical feature preview")
        cat_cols = [
            "Gender",
            "Diagnosis",
            "Treatment_Plan",
            "Symptom_1",
            "Symptom_2",
            "Symptom_3",
        ]
        for col in cat_cols:
            if col in df.columns:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head(10))

    elif page == "Model Training":
        st.header("Model Training")
        st.markdown(
            "A Decision Tree classifier is trained on the processed dataset after label encoding categorical fields."
        )
        st.subheader("Features used")
        st.write(features)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        st.metric("Training Accuracy", f"{train_acc:.2f}")
        st.metric("Test Accuracy", f"{test_acc:.2f}")

        st.subheader("Test set classification report")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.header("Predict Disease Severity")
        st.markdown(
            "Enter patient information to predict the encoded disease severity label."
        )

        with st.form(key="prediction_form"):
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox(
                "Gender", sorted(df["Gender"].astype(str).unique())
            )
            symptom_1 = st.selectbox(
                "Symptom 1", sorted(df["Symptom_1"].astype(str).unique())
            )
            symptom_2 = st.selectbox(
                "Symptom 2", sorted(df["Symptom_2"].astype(str).unique())
            )
            symptom_3 = st.selectbox(
                "Symptom 3", sorted(df["Symptom_3"].astype(str).unique())
            )
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=220, value=80)
            temperature = st.number_input("Body Temperature (C)", min_value=30.0, max_value=44.0, value=37.0)
            systolic = st.number_input("Systolic BP", min_value=0, max_value=250, value=120)
            diastolic = st.number_input("Diastolic BP", min_value=0, max_value=160, value=80)
            oxygen = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=95)
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Gender": gender,
                        "Symptom_1": symptom_1,
                        "Symptom_2": symptom_2,
                        "Symptom_3": symptom_3,
                        "Heart_Rate_bpm": heart_rate,
                        "Body_Temperature_C": temperature,
                        "Systolic_BP": systolic,
                        "Diastolic_BP": diastolic,
                        "Oxygen_Saturation_%": oxygen,
                    }
                ]
            )

            for col in ["Gender", "Symptom_1", "Symptom_2", "Symptom_3"]:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))

            input_df = input_df[features]
            prediction = model.predict(input_df)[0]
            label = encoders["Severity"].inverse_transform([prediction])[0]

            st.subheader("Prediction Result")
            st.write(f"Predicted Severity Label: **{label}**")

            st.caption("Note: The model is trained on the dataset available in the notebook and uses encoded severity labels for prediction.")

if __name__ == "__main__":
    main()
