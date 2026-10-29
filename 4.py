
import numpy as np
from collections import Counter

# Step 1: Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Step 2: KNN Classifier class
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            distances = [
                euclidean_distance(x, x_train)
                for x_train in self.X_train
            ]

            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return predictions


# Step 3: Sample Dataset
X = np.array([
    [1, 2], [2, 3], [3, 1],
    [6, 5], [7, 7], [8, 6]
])

y = np.array(['A', 'A', 'A', 'B', 'B', 'B'])


# Step 4: Test data
X_test = np.array([
    [3, 3],
    [7, 5],
    [5, 5]
])

y_test = np.array(['A', 'B', 'A'])


# Step 5: Train and Predict
knn = KNN(k=3)
knn.fit(X, y)

predictions = knn.predict(X_test)


# Step 6: Display results
print("Test Data Predictions:")

for i, (actual, predicted) in enumerate(zip(y_test, predictions)):
    status = "Correct" if actual == predicted else "Wrong"

    print(
        f"Test sample {i+1}: Actual = {actual}, "
        f"Predicted = {predicted} -> {status}"
    )
