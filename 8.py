import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# Step 1: Load Dataset
df = pd.read_csv("D:\FML python files\heart.csv")
# Step 2: Binary classification (0 = No disease, 1 = Disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
# Step 3: Prepare data
X = df.drop('target', axis=1)
y = df['target']
# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Step 6: Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
# Step 7: Predict and evaluate
y_pred = svm_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
# Step 8: Predict for a new patient
# Example input: age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal
new_patient = np.array([[57, 1, 2, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]])
new_patient_scaled = scaler.transform(new_patient)
prediction = svm_model.predict(new_patient_scaled)
print(
"\nPrediction for new patient:",
"Heart Disease" if prediction[0] == 1 else "No Heart Disease"
)
