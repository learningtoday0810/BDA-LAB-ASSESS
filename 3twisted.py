import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("countries.csv")

print("\nRaw Data:")
print(df.head())

# --------------------------------------------------
# 🔹 TWIST: Remove unwanted columns (if any)
# --------------------------------------------------
cols_to_drop = ["country", "id", "name"]
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1, errors='ignore')

# --------------------------------------------------
# 2. Select Numerical Features
# --------------------------------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])

print("\nSelected Numerical Features:")
print(numeric_df.columns)

# --------------------------------------------------
# 🔹 TWIST: Handle Missing Values
# --------------------------------------------------
print("\nMissing Values Before:\n", numeric_df.isnull().sum())

numeric_df = numeric_df.fillna(numeric_df.mean())

print("\nMissing Values After:\n", numeric_df.isnull().sum())

# --------------------------------------------------
# 🔹 TWIST: Outlier Removal (Z-Score)
# --------------------------------------------------
from scipy import stats

z = np.abs(stats.zscore(numeric_df))
numeric_df = numeric_df[(z < 3).all(axis=1)]

# --------------------------------------------------
# 🔹 TWIST: Scaling Method Choice
# --------------------------------------------------
scaler = StandardScaler()  
# scaler = MinMaxScaler()   # 👉 switch if asked

scaled_data = scaler.fit_transform(numeric_df)

# --------------------------------------------------
# 🔹 TWIST: Elbow Method (K range changeable)
# --------------------------------------------------
inertia = []

for k in range(1, 11):   # change range if needed
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Choose K dynamically (or manually)
# --------------------------------------------------
optimal_k = 3   # change if needed

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# --------------------------------------------------
# 🔹 TWIST: Add cluster labels back safely
# --------------------------------------------------
df = df.loc[numeric_df.index]
df['Cluster'] = clusters

print("\nData with Cluster Labels:")
print(df.head())

# --------------------------------------------------
# 🔹 TWIST: Cluster Analysis
# --------------------------------------------------
print("\nCluster-wise Mean Values:")
print(df.groupby('Cluster')[numeric_df.columns].mean())

# --------------------------------------------------
# 🔹 TWIST: Cluster Count
# --------------------------------------------------
print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# --------------------------------------------------
# 🔹 TWIST: 2D Visualization
# --------------------------------------------------
plt.figure()
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=clusters
)
plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])
plt.title("Country Clusters using K-Means")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Centroids Visualization
# --------------------------------------------------
centroids = kmeans.cluster_centers_

plt.figure()
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X')
plt.title("Clusters with Centroids")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Save Output
# --------------------------------------------------
df.to_csv("clustered_countries.csv", index=False)

print("\nClustering Completed Successfully")
