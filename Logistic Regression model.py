# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:31:42 2024

@author: Admin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_excel(r'D:\Warwick\Analytics in Practice\Group Assignment\filtered_columns2.xlsx')

# Preprocessing
# Drop rows with missing 'review_score' and other relevant columns
data = data.dropna(subset=['review_score'])

# Convert 'review_score' into a binary target variable
data['review_score_binary'] = data['review_score'].apply(lambda x: 1 if x >= 4 else 0)

# Encode categorical features
le_customer_state = LabelEncoder()
data['customer_state_encoded'] = le_customer_state.fit_transform(data['customer_state'])

le_payment_type = LabelEncoder()
data['payment_type_encoded'] = le_payment_type.fit_transform(data['payment_type'])

le_product_category = LabelEncoder()
data['product_category_encoded'] = le_product_category.fit_transform(data['product_category_name_english'])

# Select features and target variable
X = data[['customer_state_encoded', 'payment_type_encoded', 'payment_installments', 
          'payment_value', 'delivery_date_difference', 'product_category_encoded']]
y = data['review_score_binary']

# Handle missing values in features
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
