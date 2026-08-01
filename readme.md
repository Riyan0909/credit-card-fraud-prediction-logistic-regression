# Credit Card Fraud Detection using Logistic Regression

## Project Overview

This project uses Machine Learning to detect whether a credit card transaction is fraudulent or normal.

A Logistic Regression model is trained to classify transactions into two categories:

- 0 = Normal Transaction
- 1 = Fraud Transaction

The purpose of this project is to build a simple classification model that can identify fraudulent credit card transactions.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy

---

## Dataset

The project uses the Credit Card Fraud Detection dataset.

Dataset Features:

- Time
- V1 to V28 (Anonymized Features)
- Amount
- Class (Target Variable)

Target Column:

Class

Values:

- 0 → Normal Transaction
- 1 → Fraud Transaction

---

## Machine Learning Workflow

Steps performed in this project:

1. Loading the dataset
2. Checking dataset information
3. Separating features and target
4. Feature scaling using StandardScaler
5. Splitting data into training and testing sets
6. Training Logistic Regression model
7. Making predictions
8. Calculating accuracy score

---

## Model Used

### Logistic Regression

Logistic Regression is a classification algorithm used for predicting binary outcomes.

In this project:

- Normal Transaction = 0
- Fraud Transaction = 1

## Model Accuracy

The model performance is evaluated using Accuracy Score.

Example Output:

Model Accuracy: 99.91%
Transaction Result: Normal

## Project Structure

Credit-Card-Fraud-Detection
-credit_card.py
-creditcard.csv
-README.md
-requirements.txt

## How to Run

Install required libraries:
"pip install -r requirements.txt"

