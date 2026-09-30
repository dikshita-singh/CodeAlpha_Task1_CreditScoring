import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(df):
    # 1. Handle Missing Values
    # The Kaggle dataset has missing values in employment length and interest rate
    df = df.dropna().copy()
    
    # 2. Separate Features (X) and Target (y)
    # loan_status: 1 = Default (Bad), 0 = Non-Default (Good)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # 3. Handle Categorical Variables (Convert Text to Numbers)
    X = pd.get_dummies(X, drop_first=True)
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler