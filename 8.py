import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
df = pd.read_csv("basket.csv")

print("Dataset Head:")
print(df.head())

# Convert each row into transaction list (remove NaN values)
transactions = df.values.tolist()
transactions = [[str(item).strip() for item in row if pd.notna(item)] 
                for row in transactions]

# Apply One-Hot Encoding
encoder = TransactionEncoder()
onehot = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(onehot, columns=encoder.columns_)

print("\nOne-Hot Encoded Data Shape:", basket.shape)

frequent_itemsets = apriori(basket,
                            min_support=0.01,
                            use_colnames=True)

print("\nFrequent Itemsets Found:", frequent_itemsets.shape[0])

rules = association_rules(frequent_itemsets,
                          metric="confidence",
                          min_threshold=0.1)

rules_sorted = rules.sort_values(by="confidence", ascending=False)

top10 = rules_sorted.head(10)

print("\nTop 10 Association Rules:")
print(top10[["antecedents", "consequents",
             "support", "confidence", "lift"]])
