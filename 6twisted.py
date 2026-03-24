import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge  # Added LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import joblib

# 1. ROUND ALL FLOAT VALUES TO 2 DECIMALS
pd.options.display.float_format = '{:.2f}'.format

# 2. LOAD & INITIAL CLEANING
df = pd.read_csv("C:/Users/rajes/OneDrive/Desktop/Sem_Eight/BDA_LAB/Housing.csv")
# Drop useless columns and duplicates
cols_to_drop = ["id", "date", "serial", "row_id", "house_no"]
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1, errors='ignore')
df = df.drop_duplicates().replace(["NA", "na", "?", "NULL", " "], np.nan)

# 3. SEPARATE X AND Y (Auto-detect "Price")
target_col = [c for c in df.columns if "price" in c.lower()][0]
X = df.drop(target_col, axis=1)
y = df[target_col]

# 4. ADVANCED PREPROCESSING (The "Regex" Twist)
for col in X.columns:
    if X[col].dtype == object:
        # Extracts numbers from strings like "1200 sqft" -> 1200
        extracted = X[col].astype(str).str.extract(r"(\d+\.?\d*)")
        if extracted.notna().any().any():
            X[col] = extracted.astype(float)

# 5. UNIVERSAL MISSING & ENCODING (Fixed to include Mode)
X = X.fillna(X.mean(numeric_only=True)) # Fill numbers with mean
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna(X[col].mode()[0]) # Fill text with mode (Prevents crash)
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 6. OUTLIER REMOVAL (Twist: Z-Score)
z_scores = np.abs(stats.zscore(X.select_dtypes(include=np.number), nan_policy='omit'))
X = X[(z_scores < 3).all(axis=1)]
y = y.loc[X.index] 

# 7. SCALING, SPLIT & TRAINING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Choose Model: LinearRegression is the standard, Ridge is the Twist
model = LinearRegression() 
# model = Ridge(alpha=1.0) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. EVALUATION & VISUALIZATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_score = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2 Score: {r2:.2f}")

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"Mean CV R2 Score: {cross_val_score(model, X_scaled, y, cv=kf).mean():.2f}")

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price: Actual vs Predicted")
plt.show()

# Save the model
joblib.dump(model, "final_housing_model.pkl")
