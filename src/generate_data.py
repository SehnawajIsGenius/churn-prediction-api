import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 5000 customer records
n_samples = 5000

# Create realistic customer data
data = {
    'customer_id': range(1, n_samples + 1),
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
    'senior_citizen': np.random.choice([0, 1], n_samples),
    'partner': np.random.choice(['Yes', 'No'], n_samples),
    'dependents': np.random.choice(['Yes', 'No'], n_samples),
    'phone_service': np.random.choice(['Yes', 'No'], n_samples),
    'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples)
}

df = pd.DataFrame(data)

# Create churn based on some logic (customers with high charges and month-to-month more likely to churn)
churn_probability = (
    (df['monthly_charges'] > 70).astype(int) * 0.3 +
    (df['contract_type'] == 'Month-to-month').astype(int) * 0.4 +
    (df['tenure'] < 12).astype(int) * 0.2 +
    np.random.uniform(0, 0.1, n_samples)
)
df['churn'] = (churn_probability > 0.5).astype(int)

# Save to CSV
df.to_csv('data/raw/customer_data.csv', index=False)
print(f"✅ Generated {len(df)} customer records!")
print(f"📊 Churn rate: {df['churn'].mean():.1%}")
print(f"💾 Saved to: data/raw/customer_data.csv")
