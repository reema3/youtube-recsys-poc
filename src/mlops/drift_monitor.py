import numpy as np
import pandas as pd

def calculate_psi(expected_array, actual_array, buckets=10):
    """
    Calculates the Population Stability Index for a single feature.
    """
    # 1. Define bucket bins based on the original (expected) data distribution
    breakpoints = np.percentile(expected_array, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # 2. Count how many users fall into each bucket
    expected_counts = np.histogram(expected_array, bins=breakpoints)[0]
    actual_counts = np.histogram(actual_array, bins=breakpoints)[0]
    
    # 3. Convert counts to percentages (add a tiny fraction to avoid divide-by-zero)
    expected_percents = np.maximum(expected_counts / len(expected_array), 0.0001)
    actual_percents = np.maximum(actual_counts / len(actual_array), 0.0001)
    
    # 4. Apply the PSI formula
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    total_psi = np.sum(psi_values)
    
    return total_psi


def check_data_drift():
    print("Initiating MLOps Drift Check...")
    
    # In reality, 'expected' is your offline training lake, 
    # and 'actual' is a query from your live event stream over the last 7 days.
    
    # Simulating the original training data (mean age 25)
    training_data_ages = np.random.normal(loc=25, scale=5, size=10000)
    
    # Simulating the live production data...
    # Let's pretend a new demographic joined and the mean age shifted to 32
    live_production_ages = np.random.normal(loc=32, scale=5, size=2000)
    
    # Calculate PSI
    age_psi = calculate_psi(training_data_ages, live_production_ages)
    
    print(f"Feature 'Age' PSI Score: {age_psi:.4f}")
    
    # Programmatic Automation Rule
    if age_psi >= 0.2:
        print("CRITICAL ALERT: Severe Data Drift Detected (PSI >= 0.2).")
        print("Triggering Automated Retraining Pipeline (train.py)...")
        # Here is where you would use the `subprocess` module or an Airflow hook 
        # to automatically kick off your training script.
        return True
    elif age_psi >= 0.1:
        print("WARNING: Moderate Data Drift Detected. Monitoring closely.")
        return False
    else:
        print("SYSTEM GREEN: Data distribution is stable.")
        return False

if __name__ == "__main__":
    check_data_drift()