# brute_force.py
import itertools
import pandas as pd

# Brute Force
def brute_force(transactions, min_support, min_confidence):
    items = set(item for transaction in transactions for item in transaction)
    itemsets = []
    for i in range(1, len(items) + 1):
        itemsets.extend(itertools.combinations(items, i))
    frequent_itemsets = {}
    for itemset in itemsets:
        frequency = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
        support = frequency / len(transactions)
        if support >= min_support:
            frequent_itemsets[itemset] = support
    rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
    
    return frequent_itemsets, rules

# Function to generate association rules
def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    consequent = set(itemset) - set(antecedent)
                    antecedent_transactions = sum(1 for transaction in transactions if set(antecedent).issubset(transaction))
                    if antecedent_transactions > 0:
                        confidence = support / (antecedent_transactions / len(transactions))
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, support, confidence))
    return rules

# Function to print frequent itemsets
def print_frequent_itemsets(frequent_itemsets, method):
    if frequent_itemsets:
        itemsets_df = pd.DataFrame(
            [(set(itemset), support) for itemset, support in frequent_itemsets.items()],
            columns=['itemsets', 'support']
        )
        itemsets_df['itemsets'] = itemsets_df['itemsets'].apply(lambda x: ', '.join(x))
        print(f"{method} Frequent Itemsets:")
        print(itemsets_df.to_string(index=False))
    else:
        print(f"No frequent itemsets found using {method}.")
    print("\n")

# Function to print association rules
def print_rules(rules, method):
    if rules:
        rules_df = pd.DataFrame(rules, columns=['antecedents', 'consequents', 'support', 'confidence'])
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(x))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(x))
        print(f"{method} Association Rules:")
        print(rules_df[['antecedents', 'consequents', 'support', 'confidence']].to_string(index=False))
    else:
        print(f"No association rules found using {method}.")
    print("\n")
