import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -------------------------------------------------------------------
# 1. GLOBAL SETTINGS
# -------------------------------------------------------------------
pd.options.display.float_format = '{:.2f}'.format

# -------------------------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------------------------
df = pd.read_csv("C:/Users/rajes/OneDrive/Desktop/Sem_Eight/BDA_LAB/Titanic-Dataset_Kaggle.csv")

# -------------------------------------------------------------------
# 3. PREPROCESSING
# -------------------------------------------------------------------

# Drop unwanted columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical columns (Label Encoding first)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------------------------------------------------
# ADDING IMPORTANT TWISTS (Exam Expected)
# -------------------------------------------------------------------

# Feature Engineering: Add FamilySize and IsAlone
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# AgeGroup binning (categorical)
df['AgeGroup'] = pd.cut(df['Age'],
                        bins=[0, 12, 20, 40, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
df['AgeGroup'] = df['AgeGroup'].astype(str)
df['AgeGroup'] = le.fit_transform(df['AgeGroup'])

# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Standard Scaling (most common exam twist)
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# -------------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# -------------------------------------------------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------
# 5. MODEL TRAINING (LR + DT for comparison)
# -------------------------------------------------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree (second model to compare)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# -------------------------------------------------------------------
# 6. METRICS AND EVALUATION
# -------------------------------------------------------------------

# Logistic Regression metrics
accuracy = accuracy_score(y_test, lr_pred)
precision = precision_score(y_test, lr_pred)
recall = recall_score(y_test, lr_pred)
f1 = f1_score(y_test, lr_pred)
cm = confusion_matrix(y_test, lr_pred)

# -------------------------------------------------------------------
# 7. OUTPUT SECTION
# -------------------------------------------------------------------
print("Data Preview (First 5 rows):")
print(df.head())

print("\n--- Logistic Regression Metrics ---")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-Score  : {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# -------------------------------------------------------------------
# 8. CONFUSION MATRIX HEATMAP
# -------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# -------------------------------------------------------------------
# 9. MODEL COMPARISON
# -------------------------------------------------------------------
print("\n--- MODEL COMPARISON ---")
print("Logistic Regression Accuracy :", accuracy_score(y_test, lr_pred))
print("Decision Tree Accuracy        :", accuracy_score(y_test, dt_pred))

# -------------------------------------------------------------------
# 10. PREDICT FOR NEW PASSENGER
# -------------------------------------------------------------------
sample = np.array([[3, 1, 0, 25, 7.25,   # Pclass, SibSp, Parch, Age, Fare
                    2, 1,               # Sex_male, Sex_female (depends on encoding)
                    0, 1,               # Embarked_Q, Embarked_S
                    2, 1,               # FamilySize, IsAlone
                    2]])                # AgeGroup encoded

sample = scaler.transform(sample[:, :2]) if sample.shape[1] >= 2 else sample

print("\nPrediction for sample passenger:", lr.predict(sample))
