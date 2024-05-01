import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Definition
# Define MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Step 3: Model Training
# Train the model
mlp_model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predict on the testing set
y_pred = mlp_model.predict(X_test)

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
model_filename = "trained_models/mlp_model.pkl"
joblib.dump(mlp_model, model_filename)
print("Model saved as:", model_filename)
