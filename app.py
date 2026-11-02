import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Student Depression Classifier", layout="wide")
st.title("🎓 Student Depression Classification App")
st.write("Upload test data, select a trained model, and evaluate performance dynamically.")

# Model selector dropdown
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine")
)

model_file_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(df_test.head())

    if "Depression" not in df_test.columns:
        st.error("Uploaded CSV must contain the target column 'Depression'.")
    else:
        # Preprocessing test data
        scaler = joblib.load('model/scaler.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')

        df_proc = df_test.copy()
        y_test = df_proc['Depression'].map({'Yes': 1, 'No': 0})
        df_proc = df_proc.drop('Depression', axis=1)

        for col, le in label_encoders.items():
            if col in df_proc.columns:
                df_proc[col] = le.transform(df_proc[col])

        X_test_scaled = scaler.transform(df_proc)

        # Load selected model
        model = joblib.load(model_file_map[model_option])
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display Metrics
        st.subheader(f"Evaluation Metrics for {model_option}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # Confusion Matrix & Classification Report
        st.subheader("Confusion Matrix & Classification Report")
        c1, c2 = st.columns(2)

        with c1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        with c2:
            report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=['No', 'Yes'])
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4))
else:
    st.info("Please upload `test_data.csv` via the sidebar to execute evaluation.")