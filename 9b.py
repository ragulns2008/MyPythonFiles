import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Step 1: Load dataset
df = pd.read_csv("D:\FML python files\heart.csv")
# Step 2: Convert 'target' to binary if necessary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
# Step 3: Features and target
X = df.drop('target', axis=1)
y = df['target']
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Standardization (important for boosting, optional for RF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
boost_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
boost_model.fit(X_train_scaled, y_train)
boost_preds = boost_model.predict(X_test_scaled)
print("\nBoosting (Gradient Boosting) Results:")
print("Accuracy:", accuracy_score(y_test, boost_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, boost_preds))
