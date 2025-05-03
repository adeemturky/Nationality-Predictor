import pandas as pd
import os

# 1. Load the original dataset
df = pd.read_csv('data/final_all_names_code.csv')

# 2. Drop rows with missing names
df.dropna(subset=['Name'], inplace=True)

# 3. Mapping ISO country codes to readable nationality labels
country_mapping = {
    'AE': 'Emirati',
    'EG': 'Egyptian',
    'SA': 'Saudi',
    'JO': 'Jordanian',
    'IQ': 'Iraqi',
    'LB': 'Lebanese',
    'SY': 'Syrian',
    'YE': 'Yemeni',
    'SD': 'Sudanese',
    'PS': 'Palestinian',
    'DZ': 'Algerian',
    'MA': 'Moroccan',
    'TN': 'Tunisian',
    'IR': 'Persian',
    'TR': 'Turkish',
    'IN': 'Indian',
    'PK': 'Pakistani',
    'ID': 'Indonesian',
    'MY': 'Malaysian',
    'US': 'American',
    'GB': 'British',
    'FR': 'French',
    'DE': 'German',
    'IT': 'Italian',
    'ES': 'Spanish',
    'CN': 'Chinese',
    'JP': 'Japanese'
    # Add more if needed
}

# 4. Apply mapping and filter unmapped countries
df['origin'] = df['Country'].map(country_mapping)
df.dropna(subset=['origin'], inplace=True)

# 5. Clean names
df['name'] = df['Name'].str.lower()

# 6. Keep only needed columns
cleaned_df = df[['name', 'origin']]

# 7. Save cleaned file
os.makedirs('data/cleaned', exist_ok=True)
cleaned_df.to_csv('data/cleaned/cleaned_final_names.csv', index=False)

print("✅ Cleaned dataset saved to: data/cleaned/cleaned_final_names.csv")
print("📊 Distribution of origins:\n", cleaned_df['origin'].value_counts())
