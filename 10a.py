# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
# Step 2: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
print("K-Means Cluster Labels:\n", kmeans_labels)
print("\nK-Means Cluster Centers:\n", kmeans.cluster_centers_)
# Plot K-Means Clustering result
plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', color='red', s=100, label='Centroids')
plt.title("K-Means Clustering (First 2 Features)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend()
plt.grid(True)
plt.show()
