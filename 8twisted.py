import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Global setting
pd.options.display.float_format = '{:.2f}'.format

# Load dataset
data = pd.read_csv("C:/Users/rajes/OneDrive/Desktop/Sem_Eight/BDA_LAB/basket.csv")

print("First 5 rows:")
print(data.head())

# ------------------------------------------------------------
# 🔹 TWIST 12: Remove numeric columns
# ------------------------------------------------------------
# data = data.select_dtypes(include=['object'])

# ------------------------------------------------------------
# 🔹 TWIST 13: One-column dataset split
# ------------------------------------------------------------
# data['Items'] = data['Items'].apply(lambda x: x.split(','))
# transaction_list = data['Items'].tolist()

# ------------------------------------------------------------
# 2. Data Preprocessing
# ------------------------------------------------------------
transactions = data.apply(lambda row: row.dropna().tolist(), axis=1)
transaction_list = transactions.tolist()

# ------------------------------------------------------------
# 3. Encoding
# ------------------------------------------------------------
te = TransactionEncoder()
te_array = te.fit(transaction_list).transform(transaction_list)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# 🔹 TWIST 4: Quantity → Boolean
df_encoded = df_encoded.astype(bool)

# ------------------------------------------------------------
# 🔹 TWIST 2: Filter only 'beef'
# ------------------------------------------------------------
# df_encoded = df_encoded[df_encoded['beef'] == True]

# ------------------------------------------------------------
# 🔹 TWIST 3: Remove 'beef'
# ------------------------------------------------------------
if "beef" in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=['beef'])

# ------------------------------------------------------------
# 🔹 TWIST 14: Remove single-item transactions
# ------------------------------------------------------------
# df_encoded = df_encoded[df_encoded.sum(axis=1) > 1]

print("\nEncoded Dataset:")
print(df_encoded.head())

# ------------------------------------------------------------
# 🔹 TWIST 21: Remove rare items BEFORE Apriori
# ------------------------------------------------------------
# item_support = df_encoded.mean()
# cols = item_support[item_support > 0.01].index
# df_encoded = df_encoded[cols]

# ------------------------------------------------------------
# 🔹 TWIST 15: Limit itemset size
# ------------------------------------------------------------
frequent_itemsets = apriori(
    df_encoded,
    min_support=0.005,
    use_colnames=True,
    # max_len=2   # 👉 enable if asked
)

# 🔹 TWIST 7: Filter high-support itemsets
# frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] > 0.05]

frequent_itemsets['support'] = frequent_itemsets['support'].round(2)

# 🔹 TWIST 18: Count itemsets
print("\nTotal Frequent Itemsets:", len(frequent_itemsets))

# 🔹 TWIST 22: Top frequent itemsets
print("\nTop Frequent Itemsets:")
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

# ------------------------------------------------------------
# 🔹 TWIST 16: Use LIFT instead of confidence
# ------------------------------------------------------------
rules = association_rules(
    frequent_itemsets,
    metric="confidence",   # change to "lift" if needed
    min_threshold=0.1      # change to 1.2 if using lift
)

rules = rules.round(2)

# 🔹 TWIST 19: Count rules
print("\nTotal Rules Generated:", len(rules))

# ------------------------------------------------------------
# 🔹 TWIST 6: Consequent filter
# ------------------------------------------------------------
# rules = rules[rules['consequents'] == {'bread'}]

# ------------------------------------------------------------
# 🔹 TWIST 8: Antecedent filter
# ------------------------------------------------------------
# rules = rules[rules['antecedents'] == {'whole milk'}]

# ------------------------------------------------------------
# 🔹 TWIST 9: Lift > 1
# ------------------------------------------------------------
# rules = rules[rules['lift'] > 1]

# ------------------------------------------------------------
# 🔹 TWIST 10: Multi-condition
# ------------------------------------------------------------
# rules = rules[(rules['support'] > 0.02) & (rules['confidence'] > 0.4)]

# ------------------------------------------------------------
# 🔹 TWIST 23: Rule length filter
# ------------------------------------------------------------
# rules['length'] = rules['antecedents'].apply(lambda x: len(x)) + \
#                   rules['consequents'].apply(lambda x: len(x))
# rules = rules[rules['length'] > 2]

# ------------------------------------------------------------
# 🔹 TWIST 20: Sort by multiple columns
# ------------------------------------------------------------
# rules_sorted = rules.sort_values(by=['confidence','lift'], ascending=False)

# 🔹 TWIST 5: Sort by lift
# rules_sorted = rules.sort_values(by='lift', ascending=False)

# Default
rules_sorted = rules.sort_values(by='confidence', ascending=False)

# ------------------------------------------------------------
# 🔹 TWIST 24: Show extra metrics
# ------------------------------------------------------------
print("\nRules with Extra Metrics:")
print(rules[['antecedents','consequents','support','confidence','lift','leverage','conviction']].head())

# ------------------------------------------------------------
# 🔹 TWIST 11: Graphs
# ------------------------------------------------------------
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()

plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Lift vs Confidence")
plt.show()

plt.scatter(rules['support'], rules['lift'])
plt.xlabel("Support")
plt.ylabel("Lift")
plt.title("Lift vs Support")
plt.show()

# ------------------------------------------------------------
# Final Output
# ------------------------------------------------------------
top_10_rules = rules_sorted.head(10)

print("\nTop 10 Association Rules:")
print(top_10_rules[['antecedents','consequents','support','confidence','lift']])
