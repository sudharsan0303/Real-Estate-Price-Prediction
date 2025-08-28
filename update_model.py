import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('data/real_estate_data_cleaned.csv')

# Keep only the columns we need for the model (same as original)
df = df[['Location', 'Price', 'Area', 'BHK', 'City', 'Bathroom']]

# Fill missing Bathroom values based on BHK (number of bedrooms)
df['Bathroom'] = df['Bathroom'].fillna(df['BHK'])

# Convert column names to lowercase to match the original model
df.columns = df.columns.str.lower()

# Drop unnecessary column
if 'price_per_sqft' in df.columns:
    df = df.drop(['price_per_sqft'], axis=1)

# One-hot encode 'location' and 'city'
df = pd.get_dummies(df, columns=['location', 'city'], drop_first=False)

# Separate features and target
X = df.drop(['price'], axis=1)
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model/model_final.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature names for use during prediction
with open('model/columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and columns.pkl updated successfully.")
