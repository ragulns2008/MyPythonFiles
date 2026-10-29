
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Create Dataset (No CSV file needed)
df = pd.DataFrame({
    'Age': [22, 25, 28, 32, 35, 40, 45, 50,
            23, 30, 27, 38, 24, 42, 33, 29],
    'Income': [25000, 30000, 35000, 40000, 45000,
               65000, 70000, 80000, 28000, 38000,
               32000, 55000, 27000, 60000, 42000, 33000],
    'Credit Score': [600, 620, 650, 670, 690, 680,
                     720, 750, 590, 640, 610, 700,
                     605, 710, 675, 630],
    'Has Children': [0, 0, 1, 1, 1, 1, 1, 1,
                     0, 1, 0, 1, 0, 1, 1, 0],
    'Churn': [1, 1, 0, 0, 0, 0, 0, 0,
              1, 0, 1, 0, 1, 0, 0, 1]
})

# 2. Split Features and Target
X = df[['Age', 'Income', 'Credit Score', 'Has Children']]
y = df['Churn']

# 3. Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# 4. Train Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("--- Model Trained Successfully ---")

# 5. Evaluate Model
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Predict a New Sample
new_input = pd.DataFrame([{
    'Age': 40,
    'Income': 65000,
    'Credit Score': 680,
    'Has Children': 1
}])

prediction = model.predict(new_input)[0]

if prediction == 1:
    result = "Churn (Class 1)"
else:
    result = "No Churn (Class 0)"

print("\n--- New Sample Prediction ---")
print("Age: 40")
print("Income: 65000")
print("Credit Score: 680")
print("Has Children: 1")
print(f"Prediction: {result}")
