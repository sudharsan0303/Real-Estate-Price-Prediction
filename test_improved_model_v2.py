import pandas as pd
import numpy as np
import pickle

def prepare_input(location, city, area, bhk, bathroom, model_data):
    """Prepare input data for prediction"""
    # Create input dataframe
    input_data = pd.DataFrame({
        'Location': [location],
        'City': [city],
        'Area': [area],
        'BHK': [bhk],
        'Bathroom': [bathroom]
    })
    
    # Calculate engineered features
    input_data['room_density'] = (input_data['BHK'] / input_data['Area']).clip(0, 1)
    input_data['bath_bed_ratio'] = (input_data['Bathroom'] / input_data['BHK']).clip(0, 4)
    
    # Add price indices
    input_data['location_price_index'] = input_data['Location'].map(model_data['location_price_idx'])
    input_data['city_price_index'] = input_data['City'].map(model_data['city_price_idx'])
    
    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_data, columns=['Location', 'City'])
    
    # Ensure all columns from training are present
    for col in model_data['columns']:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Reorder columns to match training data
    input_encoded = input_encoded[model_data['columns']]
    
    # Scale numeric features
    input_encoded[model_data['numeric_features']] = model_data['scaler'].transform(
        input_encoded[model_data['numeric_features']]
    )
    
    return input_encoded

# Load the improved model
print("Loading improved model...")
with open('model/model_improved.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Test cases
test_cases = [
    {
        'location': 'Whitefield',
        'city': 'Bangalore',
        'area': 1500,
        'bhk': 3,
        'bathroom': 2,
        'description': 'Typical 3BHK in Whitefield, Bangalore'
    },
    {
        'location': 'Andheri West',
        'city': 'Mumbai',
        'area': 1000,
        'bhk': 2,
        'bathroom': 2,
        'description': 'Standard 2BHK in Andheri West, Mumbai'
    },
    {
        'location': 'Malviya Nagar',
        'city': 'Delhi',
        'area': 1200,
        'bhk': 3,
        'bathroom': 2,
        'description': 'Spacious 3BHK in Malviya Nagar, Delhi'
    }
]

print("\nTesting predictions with sample properties:")
print("===========================================")

for case in test_cases:
    print(f"\nTesting: {case['description']}")
    print(f"Location: {case['location']}")
    print(f"City: {case['city']}")
    print(f"Area: {case['area']} sq ft")
    print(f"BHK: {case['bhk']}")
    print(f"Bathrooms: {case['bathroom']}")
    
    try:
        # Prepare input and make prediction
        X = prepare_input(
            case['location'], 
            case['city'], 
            case['area'], 
            case['bhk'], 
            case['bathroom'],
            model_data
        )
        
        predicted_price = model_data['model'].predict(X)[0]
        price_per_sqft = predicted_price / case['area']
        
        print(f"Predicted Price: ₹{predicted_price:,.2f}")
        print(f"Price per sq ft: ₹{price_per_sqft:,.2f}")
    except Exception as e:
        print(f"Could not make prediction: {str(e)}")
    
# Load some actual data for comparison
df = pd.read_csv('data/real_estate_data_cleaned.csv')
print("\nActual market prices in test locations:")
print("======================================")

for case in test_cases:
    location_data = df[
        (df['Location'] == case['location']) & 
        (df['City'] == case['city']) &
        (df['BHK'] == case['bhk'])
    ]
    
    if len(location_data) > 0:
        avg_price = location_data['Price'].mean()
        avg_price_per_sqft = avg_price / case['area']
        print(f"\n{case['location']}, {case['city']} ({case['bhk']} BHK):")
        print(f"Average Market Price: ₹{avg_price:,.2f}")
        print(f"Average Price per sq ft: ₹{avg_price_per_sqft:,.2f}")
    else:
        print(f"\n{case['location']}, {case['city']} ({case['bhk']} BHK): No actual data available for comparison")
