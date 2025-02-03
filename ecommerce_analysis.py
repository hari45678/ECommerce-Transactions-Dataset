import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Load Datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Data Overview
print(customers.head())
print(products.head())
print(transactions.head())

# Exploratory Data Analysis (EDA)
## Checking missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

## Descriptive statistics
print(transactions.describe())

# Merging datasets for analysis
data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
print(data.head())

# Business Insights from EDA
## 1. Top-selling products
product_sales = data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False)
print(product_sales.head())

## 2. Revenue by Region
revenue_by_region = data.groupby('Region')['TotalValue'].sum()
print(revenue_by_region)

## 3. Customer purchasing behavior
customer_spending = data.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)
print(customer_spending.head())

## 4. Most Profitable Categories
category_revenue = data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
print(category_revenue.head())

## 5. Monthly Revenue Trends
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['Month'] = data['TransactionDate'].dt.to_period('M')
monthly_revenue = data.groupby('Month')['TotalValue'].sum()
print(monthly_revenue)

# Lookalike Model
customer_profiles = data.groupby('CustomerID').agg({'TotalValue': 'sum', 'Quantity': 'sum'})
customer_matrix = StandardScaler().fit_transform(customer_profiles)
similarity_matrix = cosine_similarity(customer_matrix)

# Getting top 3 similar customers for first 20 customers
lookalike_results = {}
for i, cust_id in enumerate(customer_profiles.index[:20]):
    sim_scores = list(enumerate(similarity_matrix[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Excluding self
    lookalike_results[cust_id] = [(customer_profiles.index[j], score) for j, score in sim_scores]

# Save Lookalike Model results
lookalike_df = pd.DataFrame.from_dict(lookalike_results, orient='index')
lookalike_df.to_csv('Lookalike.csv')

# Customer Segmentation using KMeans
customer_matrix_df = pd.DataFrame(customer_matrix, index=customer_profiles.index)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(customer_matrix)

# Merge cluster labels with original dataset
data = data.merge(pd.DataFrame({'CustomerID': customer_profiles.index, 'Cluster': kmeans_labels}), on='CustomerID')

# Evaluating clustering with DB Index
db_index = davies_bouldin_score(customer_matrix, kmeans_labels)
print(f'Davies-Bouldin Index: {db_index}')

# Visualizing Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=customer_profiles['TotalValue'], y=customer_profiles['Quantity'], hue=kmeans_labels, palette='viridis')
plt.xlabel('Total Spending')
plt.ylabel('Quantity Purchased')
plt.title('Customer Clustering')
plt.show()

# Save the processed dataset
data.to_csv('Clustered_Customers.csv', index=False)

# Generate Report
def generate_report():
    with open("Business_Insights_Report.txt", "w") as f:
        f.write("Business Insights from EDA:\n")
        f.write("1. Top-selling products:\n" + str(product_sales.head()) + "\n\n")
        f.write("2. Revenue by Region:\n" + str(revenue_by_region) + "\n\n")
        f.write("3. Customer purchasing behavior:\n" + str(customer_spending.head()) + "\n\n")
        f.write("4. Most Profitable Categories:\n" + str(category_revenue.head()) + "\n\n")
        f.write("5. Monthly Revenue Trends:\n" + str(monthly_revenue) + "\n\n")
    print("Report generated successfully.")

generate_report()
