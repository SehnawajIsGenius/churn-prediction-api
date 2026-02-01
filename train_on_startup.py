import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def train_model():
    print("🚀 Training model on server...")
    
    # Load data
    df = pd.read_csv('data/raw/customer_data.csv')
    print(f"✅ Loaded {len(df)} customer records")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 
                       'online_security', 'tech_support', 'streaming_tv',
                       'paperless_billing', 'partner', 'dependents', 
                       'phone_service', 'multiple_lines']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare data
    X = df.drop(['churn', 'customer_id'], axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training AI model...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"✅ Accuracy: {accuracy:.2%}")
    print(f"✅ AUC-ROC: {auc:.2%}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    
    print("💾 Model saved!")
    return True

if __name__ == '__main__':
    train_model() 
    