# Bank Customers Churn Prediction Using ANN 🏦📉

This repository contains a **Deep Learning project** designed to predict the likelihood of bank customers churning (closing their accounts or leaving the bank). Leveraging an **Artificial Neural Network (ANN)** built with TensorFlow and Keras, this model helps financial institutions proactively identify at-risk customers and implement data-driven retention strategies.

---

## 📌 Project Overview
Customer attrition (or churn) directly impacts a bank's profitability. It is much more expensive to acquire a new customer than to retain an existing one. This project develops an end-to-end deep learning pipeline—from data preprocessing and exploratory data analysis (EDA) to structural modeling and neural network evaluation.

### Key Features:
* **Exploratory Data Analysis (EDA):** Insights on customer demographics, balances, and behavior patterns.
* **Robust Preprocessing:** Handles categorical variables (One-Hot Encoding), feature scaling (StandardScaler), and data cleansing.
* **ANN Deep Learning Architecture:** Multi-layer Perceptron built with Keras for binary classification.
* **Performance Metrics:** Evaluated using Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curve.

---

## 📊 Dataset Description
The model is trained on a typical bank customer dataset (e.g., `Churn_Modelling.csv` from Kaggle), containing **10,000 customer records** with 14 distinct features:


| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **CreditScore** | Numerical | Customer's credit score history |
| **Geography** | Categorical | Customer's country location (e.g., France, Spain, Germany) |
| **Gender** | Categorical | Male or Female |
| **Age** | Numerical | Age of the customer |
| **Tenure** | Numerical | Number of years the customer has been with the bank |
| **Balance** | Numerical | Account balance amount |
| **NumOfProducts** | Numerical | Number of bank products used |
| **HasCrCard** | Binary | Whether the customer has a credit card (1 = Yes, 0 = No) |
| **IsActiveMember** | Binary | Whether the customer is an active member (1 = Yes, 0 = No) |
| **EstimatedSalary**| Numerical | Estimated salary of the customer |
| **Exited (Target)** | Binary | **1 if the customer churned / 0 if they stayed** |

*Note: Non-predictive identifiers like `RowNumber`, `CustomerId`, and `Surname` are dropped during preprocessing.*

---

## ⚙️ Tech Stack & Libraries
* **Language:** Python 🐍
* **Deep Learning Framework:** TensorFlow 2.x & Keras
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning Tools:** Scikit-Learn (for data splitting and scaling)

---

## 🧠 ANN Architecture Details
The core network structure includes:
1. **Input Layer:** Matches the size of the preprocessed features.
2. **Hidden Layers:** Fully connected (Dense) layers utilizing `ReLU` activation.
3. **Dropout Layers:** Applied strategically to prevent model overfitting.
4. **Output Layer:** A single node utilizing a `Sigmoid` activation function to yield a churn probability score between 0 and 1.
5. **Compilation:** Optimized using the `Adam` optimizer and tracked via `binary_crossentropy` loss.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/imshivang03/Bank_Customers_Churn_Prediction_Using_ANN.git
cd Bank_Customers_Churn_Prediction_Using_ANN
```

### 2. Install Dependencies
Make sure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Project
Open and execute the main pipeline script or Jupyter Notebook:
```bash
jupyter notebook churn_prediction.ipynb
```
*(Alternatively, run `python app.py` if structured as a standalone script).*

---

## 📈 Results & Insights
* **Accuracy:** The baseline neural network achieves an accuracy of **~85-87%** on the validation dataset.
* **Key Drivers:** Feature importance checks reveal that **Age**, **Number of Products**, and **IsActiveMember** statuses serve as major indicators determining customer churn tendencies.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to optimize the hyperparameters or try alternative network structures.

## 📄 License
This project is licensed under the MIT License.