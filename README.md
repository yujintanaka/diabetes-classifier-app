# Diabetes Prediction App Using Machine Learning

## Overview
This app predicts the likelihood of diabetes in individuals based on a 20 questions survey. It was built using Streamlit for the user interface and employs a simple Logistic Regression Classifier for predictions. The goal is to provide a simple and intuitive tool for seeing your risk profile.

---

## Features
- User-friendly interface powered by Streamlit.
- Real-time prediction based on user-provided data.
- Visual representation of your risk profile using TSNE
- Built on open-source data from Kaggle.

---

## Dataset
The dataset and survey comes from a simplified version of The Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey that is collected annually by the CDC.
The dataset used for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), containing features such as:
- Age
- Blood Pressure
- BMI
- Smoker
- Physical Activity

### Target variable:
- **Outcome**: 0 for non-diabetic, 1 for diabetic or pre-diabetic.

---

## Model Information
The machine learning model implemented in this app is a **Logistic Regression Classifier**, chosen because the model weights are small and can easily host for free, but in my experimentation the Decision Tree Classfier had the best performance.
- **Performance**: Achieved an accuracy of 75% and 91% using Logistic Regression and Random Forest. Both type 1 and type 2 errors occur at the same rate as the dataset was balanced using oversampling. From a medical ethics standpoint, it is better to have a higher rate of false positives than false negatives, so on a real production build I would make sure to use that one. You can try it for yourself if you un-comment the rfc lines.

## Classification Report (Logistic Regression)

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **0.0** | 0.75      | 0.73   | 0.74     | 43,628  |
| **1.0** | 0.74      | 0.76   | 0.75     | 43,706  |

| Metric         | Value |
|----------------|-------|
| **Accuracy**   | 0.75  |
| **Macro Avg**  | 0.75  |
| **Weighted Avg** | 0.75 |

**Total Support**: 87,334

## Classification Report (Random Forest Model):

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **0.0** | 0.96      | 0.86   | 0.91     | 43,628  |
| **1.0** | 0.88      | 0.97   | 0.92     | 43,706  |

| Metric         | Value |
|----------------|-------|
| **Accuracy**   | 0.92  |
| **Macro Avg**  | 0.92  |
| **Weighted Avg** | 0.92 |

**Total Support**: 87,334

---

## Tech Stack
- **Framework**: Streamlit
- **Programming Language**: Python
- **Libraries Used**:
  - Pandas
  - NumPy
  - Scikit-Learn

---

## Installation and Usage
### Prerequisites:
Ensure you have Python 3 installed along with the following libraries:
```bash
pip install streamlit pandas numpy scikit-learn
```

### Steps to Run:
1. Clone the repository:
   ```bash
   git clone [your-repo-link]
   ```
2. Navigate to the project directory:
   ```bash
   cd [project-directory]
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
Or, you can check it out on [Streamlit Cloud](https://diabetes-classifier-app.streamlit.app/)

Input your data into the app interface and view the prediction results instantly!

---

## Project Structure
```
├── app
│   ├── main.py              # Main application Script
├── data
│   ├── data.csv             # Pre-balanced data from kaggle used for creating small_df
│   ├── small_df.csv         # smaller version used for TSNE visuals
│   ├── train_data.csv       # Large unbalanced dataset from Kaggle using in training
├── model
│   ├── main.py              # Building the model
│   ├── model.pkl            # Trained model file
│   ├── scaler.pkl           # Scaler
├── README.md                # Project documentation
├── requirements.txt         # Dependencies list
```

## Future Scope
- Enhance model performance by experimenting with hyperparameter tuning.
- Add additional ML models with more expensive compute using a cloud platform

---

## Your Role
I handled the end-to-end development of this project, including:
- Data preprocessing and feature engineering.
- Training and optimizing the Decision Tree Classifier.
- Designing the app interface with Streamlit.

---

## Contact Information
- **GitHub**: [https://github.com/yujintanaka]
- **LinkedIn**: [https://www.linkedin.com/in/yuji-tanaka/]
- **Email**: [yntanaka2000@gmail.com]

---

Feel free to tweak the content to align with your vision! Want help polishing any sections or adding more specifics? I'm here for you! 🛠️
