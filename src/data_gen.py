import pandas as pd
import os

def load_data(file_path="data/credit_risk_dataset.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Make sure it is inside the 'data/' folder.")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df