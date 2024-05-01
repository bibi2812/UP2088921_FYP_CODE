import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Data Preparation
# Load the CSV file
data = pd.read_csv("AndroidMalwareDatasetTraining.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Labels are in the last column

# Encode class labels into numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Definition
# Define XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42)

# Step 3: Model Training
# Train the model
xgb_model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predict on the testing set
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save Model
# Create a directory for trained models if it doesn't exist
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

# Save the model as a pickle file
model_filename = "trained_models/xgb_model.pkl"
joblib.dump(xgb_model, model_filename)
print("Model saved as:", model_filename)
