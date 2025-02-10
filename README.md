# Breast Cancer Prediction using Wisconsin Dataset 

## Overview
This project is a **machine learning-based web application** for **breast cancer prediction** using the **Wisconsin Breast Cancer Dataset**. The model predicts whether a tumor is **malignant** or **benign** based on input features. The application is built using **Streamlit** for an interactive and user-friendly experience.

## Features
- **User-friendly Web Interface**: Built with Streamlit for easy interaction.
- **Machine Learning Model**: Uses algorithms like **SVM, Logistic Regression, and Decision Tree**.
- **Data Preprocessing**: Includes feature scaling and missing value handling.
- **Visualization**: Displays histograms, correlation matrices, and feature distributions.
- **Model Performance Evaluation**: Shows accuracy, precision, recall, and F1-score.

## Technologies Used
- **Python**
- **Streamlit** (for web app development)
- **Scikit-learn** (for machine learning models)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **Joblib** (for saving and loading models)

## Dataset
The **Wisconsin Breast Cancer Dataset** is used, available in the **UCI Machine Learning Repository**. It contains:
- 30 numeric features extracted from cell nuclei
- A target variable: `Malignant` (1) or `Benign` (0)

## Installation
### Step 1: Clone the Repository
```sh
git clone https://github.com/aqibfirdous/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Run the Application
```sh
streamlit run app.py
```

## Usage
1. **Upload Data (Optional)**: Use the app to upload your own CSV file.
2. **Predict Cancer Type**: Input feature values manually or use sample data.
3. **View Results**: The model predicts whether the tumor is benign or malignant.
4. **Explore Data Visualizations**: Check correlation heatmaps, histograms, and feature distributions.

## Model Training
To train the model:
```sh
python train_model.py
```
This script:
- Loads the dataset
- Preprocesses data
- Trains models (SVM, Logistic Regression, Decision Tree)
- Evaluates model performance
- Saves the best model using `joblib`

## Deployment
The app is deployed using **Streamlit Sharing** or can be hosted on **Heroku** or **AWS EC2**. To deploy:
1. **Create a Streamlit account**
2. **Push code to GitHub**
3. **Deploy via Streamlit Cloud or Heroku**

## Results & Performance
- **SVM achieved 96% accuracy**
- **Logistic Regression scored 94%**
- **Decision Tree performed at 92%**
- Evaluation metrics: **Precision, Recall, F1-score**

## Future Enhancements
- Add **deep learning** models (e.g., TensorFlow, PyTorch)
- Improve **explainability** using SHAP values
- Deploy using **Docker & Kubernetes**
- Implement **real-time prediction API**

## Contributing
Feel free to fork the repository and make improvements. Contributions are welcome!

## License
MIT License

## Contact
For any issues or improvements, open an issue in the repository or contact me at **aqibfirdous93@gmail.com**.

