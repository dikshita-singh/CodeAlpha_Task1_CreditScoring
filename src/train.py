from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model