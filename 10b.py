import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
# Step 2: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2) # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)
print("\nExplained Variance Ratio by PCA:", pca.explained_variance_ratio_)
# Plot PCA result
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='plasma', edgecolor='k', s=50)
plt.title("PCA - Iris Dataset (2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
