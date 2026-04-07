import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

print("Dataset Loaded Successfully")
print(df.head())

plt.figure()
plt.plot(df["sepallength"])
plt.title("Line Plot - Sepal Length Across Observations")
plt.xlabel("Observation Index")
plt.ylabel("Sepal Length")
plt.show()

avg_sepal = df.groupby("class")["sepallength"].mean()

plt.figure()
plt.bar(avg_sepal.index, avg_sepal.values)
plt.title("Bar Chart - Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.xticks(rotation=45)
plt.show()


plt.figure()
plt.hist(df["petallength"], bins=10)
plt.title("Histogram - Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()


plt.figure()
plt.scatter(df["sepallength"], df["petallength"])
plt.title("Scatter Plot - Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

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

print("All Graphs Generated Successfully")
