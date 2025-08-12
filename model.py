import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Load and Preprocess Dataset ------------------
# Get the absolute path to the current file (model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
file_path = os.path.join(BASE_DIR, "datasets", "loan_approval_dataset.csv")

# Load the CSV file safely
loan_data = pd.read_csv(file_path, encoding='latin1')

loan_data.columns = loan_data.columns.str.strip()
loan_data['education'] = loan_data['education'].str.strip().str.title().map({'Graduate': 1, 'Not Graduate': 0})
loan_data['self_employed'] = loan_data['self_employed'].str.strip().str.title().map({'Yes': 1, 'No': 0})
loan_data['loan_status'] = loan_data['loan_status'].str.strip().str.title().map({'Approved': 1, 'Rejected': 0})

Approved = loan_data[loan_data.loan_status == 1].sample(n=1700, random_state=42)
Rejected = loan_data[loan_data.loan_status == 0]
loan_data = pd.concat([Approved, Rejected])
x = loan_data.drop(['loan_status'], axis=1)
y = loan_data['loan_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_scaled = LogisticRegression()
model_scaled.fit(x_train_scaled, y_train)
train_accuracy_scaled = accuracy_score(model_scaled.predict(x_train_scaled), y_train)
test_accuracy_scaled = accuracy_score(model_scaled.predict(x_test_scaled), y_test)
conf_matrix = confusion_matrix(y_test, model_scaled.predict(x_test_scaled))
