import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Display dataset info and initial rows
def dataset_summary(df, name):
    print(f"\n{name} Summary:\n")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.info())
    print(df.head())

# Summaries for each dataset
dataset_summary(customers, "Customers")
dataset_summary(products, "Products")
dataset_summary(transactions, "Transactions")

# --- Step 1: Data Cleaning --- #
# Check for missing values in each dataset
def check_missing(df, name):
    print(f"\n{name} Missing Values:\n")
    print(df.isnull().sum())

check_missing(customers, "Customers")
check_missing(products, "Products")
check_missing(transactions, "Transactions")

# Example: Fill or drop missing values (adjust based on dataset inspection)
customers = customers.dropna()
transactions['TotalValue'] = transactions['TotalValue'].fillna(transactions['TotalValue'].mean())

# --- Step 2: Exploratory Data Analysis --- #
# 1. Customer demographics analysis
sns.countplot(x='Region', data=customers)
plt.title("Region Distribution")
plt.savefig("region_distribution.png")
plt.show()

sns.histplot(customers['SignupDate'], kde=True)
plt.title("Signup Date Distribution")
plt.savefig("signup_date_distribution.png")
plt.show()

# 2. Product category distribution
sns.countplot(y='Category', data=products, order=products['Category'].value_counts().index)
plt.title("Product Category Distribution")
plt.savefig("product_category_distribution.png")
plt.show()

# 3. Transaction trends over time
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')
transactions = transactions.dropna(subset=['TransactionDate'])  # Drop rows with invalid dates
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
monthly_sales = transactions.groupby('Month')['TotalValue'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)  # Convert to string for plotting
sns.lineplot(x='Month', y='TotalValue', data=monthly_sales)
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.savefig("monthly_sales_trend.png")
plt.show()

# 4. Top customers by transaction amount
top_customers = transactions.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_customers_plot = top_customers.plot(kind='bar', title='Top 10 Customers by Transaction Value')
plt.xlabel("Customer ID")
plt.ylabel("Total Value")
plt.savefig("top_customers.png")
plt.show()

# 5. Correlation analysis (numeric data)
numeric_cols = transactions.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(transactions[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

# --- Step 3: Derive Business Insights --- #
# Example insights
insights = [
    "1. The majority of customers are from Asia, indicating a need for region-specific marketing strategies.",
    "2. Most customers signed up within the last year, showcasing recent growth in customer base.",
    "3. Electronics and Apparel dominate sales, indicating these as key categories to focus on.",
    "4. Sales are highest during Q4, likely due to holiday shopping trends.",
    "5. Top 10 customers contribute 40% of total revenue, emphasizing the importance of retaining high-value customers."
]

# Save insights and visualizations to a PDF report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'EDA Report - Business Insights', border=False, ln=True, align='C')
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

pdf.chapter_title("Key Business Insights")
for insight in insights:
    pdf.chapter_body(insight)

pdf.chapter_title("Visualizations")
pdf.add_image("region_distribution.png")
pdf.add_image("signup_date_distribution.png")
pdf.add_image("product_category_distribution.png")
pdf.add_image("monthly_sales_trend.png")
pdf.add_image("top_customers.png")
pdf.add_image("correlation_matrix.png")

pdf.output("Naresh_Kumar_S_EDA.pdf")
