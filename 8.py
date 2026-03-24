import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. LOAD DATASET
df = pd.read_csv("C:/Users/rajes/OneDrive/Desktop/Sem_Eight/BDA_LAB/basket.csv")
print("Initial Data Preview:")
print(df.head())

# 2. DATA PREPROCESSING (Convert rows to transaction lists)
# We drop NaN values so that 'nan' isn't treated as a grocery item
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# 3. TRANSACTION ENCODING (One-Hot Encoding)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_).astype(bool)
print(df_encoded.head())

# 5. APPLY APRIORI ALGORITHM
# min_support defines the threshold for "frequent" items
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
frequent_itemsets['support'] = frequent_itemsets['support'].round(2)

print("\nFrequent Itemsets (Top 5):")
print(frequent_itemsets.head())

# 6. GENERATE ASSOCIATION RULES
# metric="confidence" ensures we find strong relationships
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Round numerical values for clean display
rules = rules.round(2)

# 7. SORT AND DISPLAY TOP RULES
# Sorting by confidence shows the most reliable rules first
rules_sorted = rules.sort_values(by='confidence', ascending=False)

print("\nTop 10 Association Rules:")
# Displaying only the most important columns for clarity
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
