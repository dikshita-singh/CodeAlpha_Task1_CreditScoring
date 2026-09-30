import os
import joblib
from src.data_gen import load_data
from src.preprocess import preprocess_and_split
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    print("1. Loading Kaggle Data...")
    df = load_data()
    
    print("\n2. Preprocessing Data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_and_split(df)
    
    print("\n3. Building Model...")
    model = train_model(X_train, y_train)
    
    print("\n4. Evaluating Model...")
    evaluate_model(model, X_test, y_test, feature_names)

    print("\n5. Saving Model, Scaler, and Features...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/credit_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(list(feature_names), 'models/feature_names.joblib')
    print("Success! All files saved to the 'models/' directory.")

if __name__ == "__main__":
    run_pipeline()