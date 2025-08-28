import pandas as pd

df = pd.read_csv('data/real_estate_data_filtered_final.csv')
print("🧾 Available columns:")
print(df.columns.tolist())
