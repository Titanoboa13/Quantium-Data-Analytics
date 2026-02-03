import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    # Note: Use 'QVI_transaction_data.xlsx' for your local PyCharm
    transaction_data = pd.read_excel('QVI_transaction_data.xlsx')
    customer_data = pd.read_csv('QVI_purchase_behaviour.csv')
    print("Files loaded successfully!\n")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# 2. Data Cleaning
# Convert Excel integer dates to datetime objects
transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'], unit='D', origin='1899-12-30')

# Remove the outlier customer (Loyalty Card 226000) who bought 200 packs
transaction_data = transaction_data[transaction_data['PROD_QTY'] < 200]

# Filter out non-chip products (e.g., Salsa dips)
transaction_data = transaction_data[~transaction_data['PROD_NAME'].str.lower().str.contains("salsa")]

# 3. Feature Engineering
# Extract Pack Size using raw string to avoid SyntaxWarning
transaction_data['PACK_SIZE'] = transaction_data['PROD_NAME'].str.extract(r'(\d+)').astype(float)

# Extract Brand Name
transaction_data['BRAND'] = transaction_data['PROD_NAME'].str.split().str[0]

# Clean Brand Names (Consolidating similar brands)
brand_map = {
    'Red': 'RRD', 'Smith': 'Smiths', 'Dorito': 'Doritos', 'Infzns': 'Infuzions',
    'Grain': 'GrnWves', 'Snbts': 'Sunbites', 'Natural': 'NCC', 'WW': 'Woolworths'
}
transaction_data['BRAND'] = transaction_data['BRAND'].replace(brand_map)

# 4. Data Merging
combined_data = pd.merge(transaction_data, customer_data, on='LYLTY_CARD_NBR', how='left')

# 5. Advanced Analytics
# A. Sales Summary
sales_summary = combined_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].agg(['sum', 'count']).sort_values(by='sum', ascending=False)
sales_summary.columns = ['Total_Sales', 'Transaction_Count']

# B. Average Price per Unit
combined_data['PRICE_PER_UNIT'] = combined_data['TOT_SALES'] / combined_data['PROD_QTY']
avg_price_summary = combined_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PRICE_PER_UNIT'].mean().sort_values(ascending=False)

# C. Average Units per Customer (Quantity)
units_per_customer = combined_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PROD_QTY'].sum() / \
                     combined_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].nunique()

# D. Brand Preferences for our target segment (Mainstream Young Singles/Couples)
target_segment = combined_data[(combined_data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') &
                               (combined_data['PREMIUM_CUSTOMER'] == 'Mainstream')]
brand_pref = target_segment['BRAND'].value_counts(normalize=True).head(5)

# 6. Visualizations
sns.set_style("whitegrid")

# Total Sales Plot
plt.figure(figsize=(12, 7))
sales_summary['Total_Sales'].unstack().plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Total Sales by Customer Segment')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_sales_by_segment.png')

# Avg Price Plot
plt.figure(figsize=(12, 7))
avg_price_df = avg_price_summary.reset_index()
sns.barplot(x='LIFESTAGE', y='PRICE_PER_UNIT', hue='PREMIUM_CUSTOMER', data=avg_price_df)
plt.title('Average Price per Unit by Segment')
plt.ylabel('Avg Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_price_by_segment.png')

# 7. Final Outputs
print("--- Analysis Results ---")
print("\nTop 5 Segments by Units per Customer:")
print(units_per_customer.sort_values(ascending=False).head(5))

print("\nBrand Preferences for Mainstream Young Singles/Couples:")
print(brand_pref)

# Save cleaned data for Task 2
combined_data.to_csv('QVI_data_cleaned.csv', index=False)
print("\nTask 1 Complete. 'QVI_data_cleaned.csv' has been saved.")