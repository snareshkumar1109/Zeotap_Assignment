import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# --- Step 1: Data Preparation --- #
# Merge transactions with customer data
customer_transactions = transactions.merge(customers, on='CustomerID', how='left')

# Aggregate transaction data by customer
customer_profile = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Region': 'first'  # Assuming Region is categorical
}).reset_index()

# One-hot encode categorical data (e.g., Region)
customer_profile = pd.get_dummies(customer_profile, columns=['Region'], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
numeric_cols = ['TotalValue']
customer_profile[numeric_cols] = scaler.fit_transform(customer_profile[numeric_cols])

# --- Step 2: Compute Similarities --- #
# Compute pairwise cosine similarity between customers
similarity_matrix = cosine_similarity(customer_profile[numeric_cols + list(customer_profile.columns[2:])])
similarity_df = pd.DataFrame(similarity_matrix, index=customer_profile['CustomerID'], columns=customer_profile['CustomerID'])

# --- Step 3: Generate Lookalikes --- #
# For each customer, find the top 3 similar customers
lookalike_results = {}
for customer_id in customer_profile['CustomerID']:
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:4]  # Exclude self
    lookalike_results[customer_id] = list(similar_customers.items())

# Format the results as a DataFrame
lookalike_list = []
for cust_id, similar_list in lookalike_results.items():
    for similar_cust_id, score in similar_list:
        lookalike_list.append({
            'CustomerID': cust_id,
            'SimilarCustomerID': similar_cust_id,
            'SimilarityScore': score
        })

lookalike_df = pd.DataFrame(lookalike_list)

# --- Step 4: Save Results --- #
# Save the lookalike results for the first 20 customers to Lookalike.csv
lookalike_filtered = lookalike_df[lookalike_df['CustomerID'].isin([f'C{str(i).zfill(4)}' for i in range(1, 21)])]
lookalike_filtered.to_csv('Naresh_Kumar_S_Lookalike.csv', index=False)

