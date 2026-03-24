import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Global Setting
# --------------------------------------------------
pd.options.display.float_format = '{:.2f}'.format

print("Seaborn version:", sns.__version__)

# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------
df = pd.read_csv("iris.csv")

print("\nColumns:", df.columns)
print("\nFirst 5 rows:\n", df.head())

# --------------------------------------------------
# 3. Handle Missing Values (TWIST)
# --------------------------------------------------
print("\nMissing Values Before:\n", df.isnull().sum())

# Fill numeric with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill categorical with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After:\n", df.isnull().sum())

# --------------------------------------------------
# 4. Select Numeric Columns
# --------------------------------------------------
numeric_df = df.select_dtypes(include='number')

# --------------------------------------------------
# 5. Descriptive Statistics (TWIST)
# --------------------------------------------------
print("\nDescriptive Statistics:\n", numeric_df.describe())

print("\nMedian:\n", numeric_df.median())
print("\nVariance:\n", numeric_df.var())

# --------------------------------------------------
# 6. Histogram (TWIST - custom bins)
# --------------------------------------------------
plt.figure(figsize=(10, 6))
numeric_df.hist(bins=15)
plt.suptitle("Histogram of Iris Features")
plt.show()

# --------------------------------------------------
# 7. Box Plot (TWIST - by category)
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[df.columns[-1]], y=df[numeric_df.columns[0]])
plt.title("Box Plot by Species")
plt.xticks(rotation=45)
plt.show()

# --------------------------------------------------
# 8. Correlation Heatmap
# --------------------------------------------------
corr = numeric_df.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------------
# 9. Scatter Plot (TWIST - with hue)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=numeric_df.columns[0],
    y=numeric_df.columns[2],
    hue=df[df.columns[-1]],
    data=df
)
plt.title("Scatter Plot with Species")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Pairplot (VERY IMPORTANT)
# --------------------------------------------------
sns.pairplot(df, hue=df.columns[-1])
plt.show()

# --------------------------------------------------
# 10. Feature-wise Distribution (Loop)
# --------------------------------------------------
for column in numeric_df.columns:
    plt.figure(figsize=(8, 3))

    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(numeric_df[column], kde=True)
    plt.title(f"Histogram of {column}")

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=numeric_df[column])
    plt.title(f"Box Plot of {column}")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 🔹 TWIST: GroupBy Analysis
# --------------------------------------------------
print("\nAverage Values by Species:\n")
print(df.groupby(df.columns[-1]).mean(numeric_only=True))

# --------------------------------------------------
# 🔹 TWIST: Save Plot
# --------------------------------------------------
plt.figure()
sns.histplot(numeric_df[numeric_df.columns[0]], kde=True)
plt.title("Saved Plot Example")
plt.savefig("iris_plot.png")

print("\nEDA Completed Successfully")
