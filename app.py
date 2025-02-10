import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# Load Models
def load_model(model_name):
    if model_name == 'SVM':
        return joblib.load("svm_model.pkl")
    elif model_name == 'Logistic Regression':
        return joblib.load("logistic_regression_model.pkl")
    elif model_name == 'Decision Tree':
        return joblib.load("decision_tree_model.pkl")


# Title and Description
st.title("Breast Cancer Prediction App")
st.write("Select a model and provide input parameters for prediction of tumor malignancy.")

# Model Selection
model_choice = st.selectbox("Choose a model", ("SVM", "Logistic Regression", "Decision Tree"))
model = load_model(model_choice)

# Input Features
radius_mean = st.number_input("Radius Mean", min_value=0.0, value=10.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0, value=10.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=100.0)
area_mean = st.number_input("Area Mean", min_value=0.0, value=500.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.1)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.1)
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.1)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.1)


# Arrange input features in the correct order expected by the model
input_features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                            smoothness_mean, compactness_mean, concavity_mean,
                            concave_points_mean, symmetry_mean, fractal_dimension_mean]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    st.write("Prediction:", diagnosis)

    # Mock true label for testing (replace with actual label if available)
    y_true = np.array([1])  # Assuming true label is Malignant for testing
    y_pred = prediction

    # Handle case where only one class is predicted
    if len(np.unique(y_pred)) == 1:
        st.write("Target class predicton (Malignant or Benign)")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], labels=[0, 1])
        st.subheader("Classification Report")
        st.text(report)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # ROC Curve
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, model.predict_proba(input_features)[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot()
