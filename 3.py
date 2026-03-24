
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("countries.csv")   

print("\n Raw Data:")
print(df.head())

numeric_df = df.select_dtypes(include=['numbers'])

print("\n Selected Numerical Features:")
print(numeric_df.columns)


numeric_df = numeric_df.fillna(numeric_df.mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters

print("\n Data with Cluster Labels:")
print(df.head())

print("\n Cluster-wise Mean Values:")
print(df.groupby('Cluster')[numeric_df.columns].mean())

plt.figure()
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters)
plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])
plt.title("Country Clusters using K-Means")
plt.show()
