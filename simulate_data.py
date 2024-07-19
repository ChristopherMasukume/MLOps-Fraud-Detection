import pandas as pd
import numpy as np
import os

def simulate_monthly_data(base_data_path, output_dir, months=12):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_data = pd.read_csv(base_data_path)
    for month in range(1, months + 1):
        new_data = base_data.copy()
        drift_factor = np.random.normal(0, 0.1, new_data.shape)
        new_data += drift_factor
        new_data_path = os.path.join(output_dir, f'creditcard_month_{month}.csv')
        new_data.to_csv(new_data_path, index=False)
        print(f"Simulated data for month {month} saved to {new_data_path}")

if __name__ == "__main__":
    simulate_monthly_data('C:/Fraud_Detection/creditcard.csv', 'C:/Fraud_Detection/simulated_data')

#'C:/Fraud_Detection/creditcard.csv', 