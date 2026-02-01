import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import json

print("🚀 Starting model training...")

# Load data
df = pd.read_csv('data/raw/customer_data.csv')
print(f"✅ Loaded {len(df)} customer records")

# Prepare data
print("🔧 Preparing data...")

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

# Separate features and target
X = df.drop(['churn', 'customer_id'], axis=1)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {len(X_train)} records")
print(f"📊 Test set: {len(X_test)} records")

# Train model
print("🤖 Training AI model...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*50)
print("📈 MODEL PERFORMANCE")
print("="*50)
print(f"✅ Accuracy:  {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall:    {recall:.2%}")
print(f"✅ AUC-ROC:   {auc:.2%}")
print("="*50)

# Save model
joblib.dump(model, 'models/model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'auc_roc': float(auc)
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n💾 Model saved to: models/model.pkl")
print("✅ Training complete!")
