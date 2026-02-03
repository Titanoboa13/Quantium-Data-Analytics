import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
data = pd.read_csv('QVI_data.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEARMONTH'] = data['DATE'].dt.strftime('%Y%m').astype(int)


# 2. Monthly metrics function
def get_monthly_metrics(df):
    monthly_stats = df.groupby(['STORE_NBR', 'YEARMONTH']).agg(
        totSales=('TOT_SALES', 'sum'),
        nCustomers=('LYLTY_CARD_NBR', 'nunique'),
        nTransactions=('TXN_ID', 'nunique'),
        totQty=('PROD_QTY', 'sum')
    ).reset_index()
    monthly_stats['nTxnPerCust'] = monthly_stats['nTransactions'] / monthly_stats['nCustomers']
    monthly_stats['avgPricePerUnit'] = monthly_stats['totSales'] / monthly_stats['totQty']
    return monthly_stats


measure_over_time = get_monthly_metrics(data)
pre_trial_measures = measure_over_time[measure_over_time['YEARMONTH'] < 201902]


# 3. Functions for Store Matching
def calculate_correlation(input_table, metric_col, trial_store):
    pivot_table = input_table.pivot(index='YEARMONTH', columns='STORE_NBR', values=metric_col)
    return pivot_table.corrwith(pivot_table[trial_store])


def calculate_magnitude_distance(input_table, metric_col, trial_store):
    pivot_table = input_table.pivot(index='YEARMONTH', columns='STORE_NBR', values=metric_col)
    abs_diff = abs(pivot_table.subtract(pivot_table[trial_store], axis=0))
    mag_score = 1 - (abs_diff.subtract(abs_diff.min(axis=1), axis=0).divide(
        abs_diff.max(axis=1) - abs_diff.min(axis=1), axis=0))
    return mag_score.mean()


def find_control_store(trial_store_id, pre_trial_data):
    corr_sales = calculate_correlation(pre_trial_data, 'totSales', trial_store_id)
    mag_sales = calculate_magnitude_distance(pre_trial_data, 'totSales', trial_store_id)
    corr_cust = calculate_correlation(pre_trial_data, 'nCustomers', trial_store_id)
    mag_cust = calculate_magnitude_distance(pre_trial_data, 'nCustomers', trial_store_id)
    scores = pd.DataFrame({'corr_sales': corr_sales, 'mag_sales': mag_sales,
                           'corr_cust': corr_cust, 'mag_cust': mag_cust})
    scores['final_score'] = scores.mean(axis=1)
    return scores.sort_values(by='final_score', ascending=False).drop(trial_store_id)


# 4. Define Trial-Control Pairs
trial_control_map = {77: 233, 86: 155, 88: 237}

# 5. Assessment and Visualization
trial_period = [201902, 201903, 201904]

for ts, cs in trial_control_map.items():
    # Calculate Scaling Factor based on Pre-trial Sales
    pre_trial_ts_sales = pre_trial_measures[pre_trial_measures['STORE_NBR'] == ts]['totSales'].mean()
    pre_trial_cs_sales = pre_trial_measures[pre_trial_measures['STORE_NBR'] == cs]['totSales'].mean()
    scaling_factor = pre_trial_ts_sales / pre_trial_cs_sales

    # Apply scaling to Control Store Sales
    mask = measure_over_time['STORE_NBR'] == cs
    measure_over_time.loc[mask, 'scaledSales'] = measure_over_time.loc[mask, 'totSales'] * scaling_factor

    # Create the Visual Comparison
    plt.figure(figsize=(10, 5))
    ts_data = measure_over_time[measure_over_time['STORE_NBR'] == ts]
    cs_data = measure_over_time[measure_over_time['STORE_NBR'] == cs]

    plt.plot(ts_data['YEARMONTH'].astype(str), ts_data['totSales'], label=f'Trial Store {ts}', marker='o', linewidth=2)
    plt.plot(cs_data['YEARMONTH'].astype(str), cs_data['scaledSales'], label=f'Control Store {cs} (Scaled)',
             linestyle='--', marker='x')
    plt.axvspan('201902', '201904', color='gray', alpha=0.2, label='Trial Period')

    plt.title(f'Performance Comparison: Store {ts} vs Control {cs}', fontsize=12)
    plt.ylabel('Total Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'trial_comparison_{ts}.png')
    print(f"Graph for Store {ts} saved as trial_comparison_{ts}.png")