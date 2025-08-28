import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the model and columns
with open('model/model_final.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# Load the test data (using a portion of the cleaned dataset)
df = pd.read_csv('data/real_estate_data_cleaned.csv')
df = df[['Location', 'Price', 'Area', 'BHK', 'City', 'Bathroom']]

# Fill missing Bathroom values based on BHK (number of bedrooms)
df['Bathroom'] = df['Bathroom'].fillna(df['BHK'])

df.columns = df.columns.str.lower()

# Create test set (using 20% of the data)
test_size = int(len(df) * 0.2)
test_df = df.sample(n=test_size, random_state=42)

# Prepare test features
X_test = test_df.drop('price', axis=1)
y_test = test_df['price']

# One-hot encode the test data
X_test_encoded = pd.get_dummies(X_test, columns=['location', 'city'], drop_first=False)

# Align test features with training features
X_test_aligned = pd.DataFrame(columns=columns)
for col in columns:
    if col in X_test_encoded.columns:
        X_test_aligned[col] = X_test_encoded[col]
    else:
        X_test_aligned[col] = 0

# Make predictions
y_pred = model.predict(X_test_aligned)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance Metrics ===")
print(f"Root Mean Square Error: ₹{rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Show some sample predictions
print("\n=== Sample Predictions ===")
sample_size = min(5, len(test_df))
samples = test_df.sample(n=sample_size, random_state=42)

for idx, row in samples.iterrows():
    actual_price = row['price']
    
    # Prepare single sample for prediction
    sample = row.drop('price').to_frame().T
    sample_encoded = pd.get_dummies(sample, columns=['location', 'city'], drop_first=False)
    
    # Align features
    sample_aligned = pd.DataFrame(columns=columns)
    for col in columns:
        if col in sample_encoded.columns:
            sample_aligned[col] = sample_encoded[col]
        else:
            sample_aligned[col] = 0
    
    predicted_price = model.predict(sample_aligned)[0]
    
    print(f"\nProperty Details:")
    print(f"Location: {row['location']}")
    print(f"City: {row['city']}")
    print(f"Area: {row['area']} sq ft")
    print(f"BHK: {row['bhk']}")
    print(f"Bathrooms: {row['bathroom']}")
    print(f"Actual Price: ₹{actual_price:,.2f}")
    print(f"Predicted Price: ₹{predicted_price:,.2f}")
    print(f"Difference: {abs(actual_price - predicted_price) / actual_price * 100:.2f}%")
