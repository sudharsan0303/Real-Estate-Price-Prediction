import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load and prepare the data
print("Loading and preparing data...")
df = pd.read_csv('data/real_estate_data_cleaned.csv')

# Keep only the necessary columns and handle missing values
df = df[['Location', 'Price', 'Area', 'BHK', 'City', 'Bathroom']].copy()
df['Bathroom'] = df['Bathroom'].fillna(df['BHK'])

# Drop any remaining rows with missing values
df = df.dropna()

print("Performing feature engineering...")

# Calculate price per square foot
df['price_per_sqft'] = df['Price'] / df['Area']

# Create location and city price indices
location_price_idx = df.groupby('Location')['price_per_sqft'].median().to_dict()
city_price_idx = df.groupby('City')['price_per_sqft'].median().to_dict()

df['location_price_index'] = df['Location'].map(location_price_idx)
df['city_price_index'] = df['City'].map(city_price_idx)

# Create room density and bathroom ratio features
df['room_density'] = (df['BHK'] / df['Area']).clip(0, 1)
df['bath_bed_ratio'] = (df['Bathroom'] / df['BHK']).clip(0, 4)

# 4. Create location price indicators
location_mean_prices = df.groupby('Location')['price_per_sqft'].mean()
df['location_price_index'] = df['Location'].map(location_mean_prices)

# 5. Create city price indicators
city_mean_prices = df.groupby('City')['price_per_sqft'].mean()
df['city_price_index'] = df['City'].map(city_mean_prices)

# Prepare features for modeling
features = ['Area', 'BHK', 'Bathroom', 'room_density', 'bath_bed_ratio', 
           'location_price_index', 'city_price_index', 'Location', 'City']
target = 'Price'

# Split the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle categorical variables with one-hot encoding
X = pd.get_dummies(X, columns=['Location', 'City'], drop_first=False)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

# Scale numerical features
numeric_features = ['Area', 'BHK', 'Bathroom', 'room_density', 'bath_bed_ratio', 
                   'location_price_index', 'city_price_index']
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train the model
print("Training the model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model performance...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\nTraining Set Performance:")
print(f"RMSE: ₹{train_rmse:,.2f}")
print(f"R² Score: {train_r2:.4f}")

print(f"\nTest Set Performance:")
print(f"RMSE: ₹{test_rmse:,.2f}")
print(f"R² Score: {test_r2:.4f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the improved model and preprocessing components
print("\nSaving the model and preprocessing components...")
with open('model/model_improved.pkl', 'wb') as f:
    model_data = {
        'model': model,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'columns': X_train.columns.tolist(),
        'location_price_idx': location_price_idx,
        'city_price_idx': city_price_idx
    }
    pickle.dump(model_data, f)

print("✅ Improved model saved successfully.")
