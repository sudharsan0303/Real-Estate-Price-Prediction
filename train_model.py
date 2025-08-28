# train_model.py
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/real_estate_data_filtered_final.csv')

X = df[['Location', 'City', 'Area', 'BHK', 'Bathroom']]
y = df['Price']

X_processed = pd.get_dummies(X)
model = LinearRegression()
model.fit(X_processed, y)

# Save model
with open('model/model_final.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save columns
with open('model/columns.pkl', 'wb') as f:
    pickle.dump(X_processed.columns.tolist(), f)

print("✅ Model and columns saved successfully.")
