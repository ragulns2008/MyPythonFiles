import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Step 1: Load Dataset from CSV file
# Make sure 'dataset.csv' is in the same directory as this script
df = pd.read_csv("D:\FML python files\heart.csv")
# Extract features and labels from the DataFrame
# Assuming column names are 'Hours_Studied', 'Hours_Slept', and 'Label'
X = df[['Hours Studied', 'Hours Slept']].values
y = df['Label'].values
# Step 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Step 3: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Step 4: Predict
y_pred = model.predict(X_test)
# Step 5: Evaluate
print("Predicted:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Step 6: Predict a new sample
new_sample = np.array([[4, 7]]) # Example: 4 hours study, 7 hours sleep
new_pred = model.predict(new_sample)
print("\nPrediction for [4 hrs study, 7 hrs sleep]:", "Pass" if new_pred[0] == 1 else "Fail")
