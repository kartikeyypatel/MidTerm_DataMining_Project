import csv
import itertools
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import pandas as pd
from brute_force_module import brute_force, print_frequent_itemsets, print_rules

def read_data(filename):
    transactions = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            transactions.append(row[1].split(", "))
    return transactions

def run_apriori_fpgrowth(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    #Apriori
    
    start_time = time.time()
    frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
    end_time = time.time()
    print(f"Apriori Execution Time: {end_time - start_time} seconds \n")
    print("Apriori Frequent Itemsets")
    print(frequent_itemsets_apriori, "\n")
    if not rules_apriori.empty:
        print("Apriori Association Rules : ")
        # Convert antecedents and consequents to strings without square brackets
        rules_apriori['antecedents'] = rules_apriori['antecedents'].apply(lambda x: ', '.join(x))
        rules_apriori['consequents'] = rules_apriori['consequents'].apply(lambda x: ', '.join(x))
        print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence']].head(10).to_string(index=False))
    else:
        print("No association rules found.")
    print("\n")

    # FP-Growth
    
    start_time = time.time()
    frequent_itemsets_fp = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)
    end_time = time.time()
    print(f"FP-Growth Execution Time: {end_time - start_time} seconds \n")
    print("FP-Growth Frequent Itemsets")
    print(frequent_itemsets_fp, "\n")
    if not rules_fp.empty:
        print("FP-Growth Association Rules : ")
        # Convert antecedents and consequents to strings without square brackets
        rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join(x))
        rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join(x))
        print(rules_fp[['antecedents', 'consequents', 'support', 'confidence']].head(10).to_string(index=False))
    else:
        print("No association rules found.")
    print("\n")

def main():
    datasets = ["amazon.csv", "BestBuy.csv", "Kmart.csv", "nike.csv", "Generic.csv"]
    dataset_choice = int(input("Choose a dataset (1-5): \n1.Amazon\n2.BestBuy\n3.Kmart\n4.Nike\n5.Generic\n"))
    min_support = float(input("Enter minimum support (as a decimal): "))
    min_confidence = float(input("Enter minimum confidence (as a decimal): "))

    transactions = read_data(datasets[dataset_choice - 1])

    # Brute Force
    start_time = time.time()
    frequent_itemsets, rules = brute_force(transactions, min_support, min_confidence)
    end_time = time.time()
    print(f"\nBrute Force Execution Time: {end_time - start_time} seconds \n")
    print_frequent_itemsets(frequent_itemsets, method="Brute Force")
    print_rules(rules, method="Brute Force")


    # Apriori and FP-Growth
    run_apriori_fpgrowth(transactions, min_support, min_confidence)

if __name__ == "__main__":
    main()