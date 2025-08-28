import pandas as pd
import numpy as np

# Read the merged dataset
df = pd.read_csv('data/merged_real_estate_data.csv')

# Handle missing values for each column
def handle_missing_data(df):
    # 1. Status - Fill with 'Unknown' or most common value
    df['Status'] = df['Status'].fillna(df['Status'].mode()[0] if not df['Status'].mode().empty else 'Unknown')
    
    # 2. Furnishing - Fill with 'Unfurnished' as default
    df['Furnishing'] = df['Furnishing'].fillna('Unfurnished')
    
    # 3. Parking - Fill with 0 (assuming no parking)
    df['Parking'] = df['Parking'].fillna(0)
    
    # 4. Transaction - Fill with 'New Property' as default
    df['Transaction'] = df['Transaction'].fillna('New Property')
    
    # 5. Property_Type - Fill with 'Apartment' as default
    df['Property_Type'] = df['Property_Type'].fillna('Apartment')
    
    return df

# Clean the data
df_cleaned = handle_missing_data(df)

# Calculate Price per Square Foot
df_cleaned['Price_per_Sqft'] = df_cleaned['Price'] / df_cleaned['Area']

# Save the cleaned dataset
df_cleaned.to_csv('data/real_estate_data_cleaned.csv', index=False)

# Print statistics
print("\nOriginal Dataset Stats:")
print(df.info())
print("\nCleaned Dataset Stats:")
print(df_cleaned.info())
