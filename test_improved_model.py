import pandas as pd
import numpy as np
import pickle

# Load the improved model and components
print("Loading improved model...")
with open('model/model_improved.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    numeric_features = model_data['numeric_features']
    columns = model_data['columns']

def prepare_sample(location, city, area, bhk, bathroom):
    """Prepare a single sample for prediction"""
    # Create a sample DataFrame
    sample = pd.DataFrame({
        'Area': [area],
        'BHK': [bhk],
        'Bathroom': [bathroom],
        'Location': [location],
        'City': [city]
    })
    
    # Calculate engineered features
    sample['room_density'] = sample['BHK'] / sample['Area']
    sample['bath_bed_ratio'] = sample['Bathroom'] / sample['BHK']
    
    # Load the full dataset to get price indices
    df = pd.read_csv('data/real_estate_data_cleaned.csv')
    
    # Calculate location and city price indices
    location_mean_prices = df.groupby('Location')['Price'].mean() / df.groupby('Location')['Area'].mean()
    city_mean_prices = df.groupby('City')['Price'].mean() / df.groupby('City')['Area'].mean()
    
    sample['location_price_index'] = sample['Location'].map(location_mean_prices)
    sample['city_price_index'] = sample['City'].map(city_mean_prices)
    
    # One-hot encode categorical variables
    sample_encoded = pd.get_dummies(sample, columns=['Location', 'City'])
    
    # Ensure all columns from training are present
    for col in columns:
        if col not in sample_encoded.columns:
            sample_encoded[col] = 0
    
    # Reorder columns to match training data
    sample_encoded = sample_encoded[columns]
    
    # Scale numeric features
    sample_encoded[numeric_features] = scaler.transform(sample_encoded[numeric_features])
    
    return sample_encoded

# Test cases
test_cases = [
    {
        'location': 'Whitefield',
        'city': 'Bangalore',
        'area': 1500,
        'bhk': 3,
        'bathroom': 2
    },
    {
        'location': 'Andheri West',
        'city': 'Mumbai',
        'area': 1000,
        'bhk': 2,
        'bathroom': 2
    },
    {
        'location': 'Malviya Nagar',
        'city': 'Delhi',
        'area': 1200,
        'bhk': 3,
        'bathroom': 2
    }
]

print("\nTesting predictions with sample properties:")
print("===========================================")

for case in test_cases:
    # Prepare the sample
    X = prepare_sample(**case)
    
    # Make prediction
    predicted_price = model.predict(X)[0]
    
    print(f"\nProperty Details:")
    print(f"Location: {case['location']}")
    print(f"City: {case['city']}")
    print(f"Area: {case['area']} sq ft")
    print(f"BHK: {case['bhk']}")
    print(f"Bathrooms: {case['bathroom']}")
    print(f"Predicted Price: ₹{predicted_price:,.2f}")
    
    # Calculate price per sq ft
    price_per_sqft = predicted_price / case['area']
    print(f"Price per sq ft: ₹{price_per_sqft:,.2f}")
