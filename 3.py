import numpy as np
import pandas as pd
# Step 1: Sample dataset
data = {
'Size_sqft': [600, 800, 1000, 1200, 1400, 1600],
'Bedrooms': [1, 2, 2, 3, 3, 4],
'Rent': [10000, 15000, 18000, 22000, 25000, 28000]
}
df = pd.DataFrame(data)
# Step 2: Prepare features (X) and target (y)
X = df[['Size_sqft', 'Bedrooms']].values
y = df['Rent'].values.reshape(-1, 1)
X_b = np.hstack([np.ones((X.shape[0], 1)), X]) # Add intercept column
# Step 3: Calculate theta using Normal Equation: theta = (XᵀX)⁻¹ Xᵀy
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# Step 4: Print model parameters
intercept = theta[0][0]
coefficients = theta[1:].flatten()
print("Intercept (base rent):", round(intercept))
print("Coefficients:", [round(c) for c in coefficients])
# Step 5: Predict for a new input (1100 sqft, 2 BHK)
new_data = np.array([[1, 1100, 2]]) # 1 for intercept
predicted_rent = new_data.dot(theta)
print("Predicted Rent for 1100 sqft, 2 BHK:", round(predicted_rent[0][0]))
