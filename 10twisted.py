import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("iris.csv")

print("Dataset Loaded Successfully")
print(df.head())

# --------------------------------------------------
# 🔹 TWIST: Handle Missing Values
# --------------------------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --------------------------------------------------
# 🔹 TWIST: Select specific columns (if needed)
# --------------------------------------------------
# df = df[["sepallength", "sepalwidth", "petallength", "petalwidth", "class"]]

# --------------------------------------------------
# 2. Line Plot
# --------------------------------------------------
plt.figure()
plt.plot(df["sepallength"], marker='o')   # TWIST: marker added
plt.title("Line Plot - Sepal Length Across Observations")
plt.xlabel("Observation Index")
plt.ylabel("Sepal Length")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Line plot for another column
# --------------------------------------------------
# plt.plot(df["petallength"])

# --------------------------------------------------
# 3. Bar Chart
# --------------------------------------------------
avg_sepal = df.groupby("class")["sepallength"].mean()

plt.figure()
plt.bar(avg_sepal.index, avg_sepal.values)
plt.title("Bar Chart - Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.xticks(rotation=45)
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Bar chart for another feature
# --------------------------------------------------
# avg_petal = df.groupby("class")["petallength"].mean()

# --------------------------------------------------
# 4. Histogram
# --------------------------------------------------
plt.figure()
plt.hist(df["petallength"], bins=15)   # TWIST: bins changed
plt.title("Histogram - Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Histogram without bins
# --------------------------------------------------
# plt.hist(df["sepallength"])

# --------------------------------------------------
# 5. Scatter Plot
# --------------------------------------------------
plt.figure()
plt.scatter(df["sepallength"], df["petallength"])
plt.title("Scatter Plot - Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Scatter with color based on class
# --------------------------------------------------
# colors = df["class"].astype('category').cat.codes
# plt.scatter(df["sepallength"], df["petallength"], c=colors)

# --------------------------------------------------
# 6. Box Plot
# --------------------------------------------------
species = df["class"].unique()

data_to_plot = [
    df[df["class"] == sp]["sepalwidth"]
    for sp in species
]

plt.figure()
plt.boxplot(data_to_plot)
plt.title("Box Plot - Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width")
plt.xticks(range(1, len(species)+1), species, rotation=45)
plt.show()

# --------------------------------------------------
# 🔹 TWIST: Boxplot for another feature
# --------------------------------------------------
# data_to_plot = [df[df["class"] == sp]["petalwidth"] for sp in species]

# --------------------------------------------------
# 🔹 TWIST: Save graph
# --------------------------------------------------
plt.figure()
plt.hist(df["sepallength"])
plt.title("Saved Plot Example")
plt.savefig("sepal_plot.png")

# --------------------------------------------------
# 🔹 TWIST: GroupBy analysis
# --------------------------------------------------
print("\nMean values by species:")
print(df.groupby("class").mean(numeric_only=True))

# --------------------------------------------------
# 🔹 TWIST: Sort values
# --------------------------------------------------
print("\nTop 5 Highest Sepal Length:")
print(df.sort_values(by="sepallength", ascending=False).head())

print("\nAll Graphs Generated Successfully")
