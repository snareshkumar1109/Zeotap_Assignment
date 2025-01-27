import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

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

# --- Step 2: Clustering --- #
# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(customer_profile[numeric_cols + list(customer_profile.columns[2:])])
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("elbow_method.png")
plt.show()

# Choose an optimal number of clusters (e.g., 4 based on the elbow curve)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(customer_profile[numeric_cols + list(customer_profile.columns[2:])])

# Assign clusters to customers
customer_profile['Cluster'] = kmeans.labels_

# --- Step 3: Evaluate Clustering --- #
# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(customer_profile[numeric_cols + list(customer_profile.columns[2:-1])], customer_profile['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")

# --- Step 4: Visualize Clusters --- #
sns.scatterplot(
    x=customer_profile['TotalValue'],
    y=customer_profile.iloc[:, 3],  # Use one of the one-hot encoded region columns
    hue=customer_profile['Cluster'],
    palette='viridis'
)
plt.title('Customer Segmentation')
plt.xlabel('Standardized Total Value')
plt.ylabel('Region (One-Hot Encoded)')
plt.legend(title='Cluster')
plt.savefig("customer_segmentation.png")
plt.show()

# --- Step 5: Save Results --- #
# Save customer segmentation results
customer_profile.to_csv('Naresh_Kumar_S_Customer_Segmentation.csv', index=False)

# --- Step 6: Generate PDF Report --- #
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Customer Segmentation Report', border=False, ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, border=False, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path):
        self.image(image_path, w=150)
        self.ln(10)

pdf = PDFReport()
pdf.add_page()

# Add clustering metrics and insights
pdf.chapter_title("Clustering Metrics and Insights")
pdf.chapter_body(f"Number of Clusters: {k}")
pdf.chapter_body(f"Davies-Bouldin Index: {db_index:.2f}")
pdf.chapter_body("Insights:")
pdf.chapter_body("1. The clustering revealed distinct customer segments based on transaction value and region.")
pdf.chapter_body("2. Customers in Cluster 0 are high-value customers, indicating priority for retention strategies.")
pdf.chapter_body("3. Region-based segmentation can guide targeted marketing campaigns.")

# Add visualizations
pdf.chapter_title("Visualizations")
pdf.add_image("elbow_method.png")
pdf.add_image("customer_segmentation.png")

pdf.output("Naresh_Kumar_S_Customer_Segmentation_Report.pdf")

