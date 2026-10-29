import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# 1. Load the dataset
data = load_iris()
X = data.data
y = data.target
# 2. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Initialize and train the Gaussian Naïve Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)
# 4. Make predictions
y_pred = model.predict(X_test)
# 5. Output the Results
print("--- Naïve Bayes Classification Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
# 6. Classify a Single New Sample
# Example: Sepal Length, Sepal Width, Petal Length, Petal Width
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_sample)
print(f"\nNew Sample Prediction: {data.target_names[prediction][0]}")
