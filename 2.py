

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

 

print("Seaborn version:", sns.__version__)

 

# Load dataset

df = pd.read_csv("iris.csv")

 

# Display column names (for confirmation)

print("\nColumns:", df.columns)

 

# Select only numerical columns (SAFE METHOD)

numeric_df = df.select_dtypes(include='number')

 

# 1. Check missing values

print("\nMissing Values:\n", df.isnull().sum())

 

# 2. Descriptive statistics

print("\nDescriptive Statistics:\n", numeric_df.describe())

 

# 3. Histogram

numeric_df.hist(figsize=(10, 8))

plt.suptitle("Histogram of Iris Features")

plt.show()

 

# 4. Box Plot

plt.figure(figsize=(8, 5))

sns.boxplot(data=numeric_df)

plt.title("Box Plot of Iris Features")

plt.show()

 

# 5. Correlation Heatmap

corr = numeric_df.corr()

plt.figure(figsize=(7, 5))

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

 

# 6. Feature-wise Distribution

for column in numeric_df.columns:

    plt.figure(figsize=(8, 3))

 

    plt.subplot(1, 2, 1)

    sns.histplot(numeric_df[column], kde=True)

    plt.title(f"Histogram of {column}")

 

    plt.subplot(1, 2, 2)

    sns.boxplot(y=numeric_df[column])

    plt.title(f"Box Plot of {column}")

 

    plt.tight_layout()

    plt.show()


