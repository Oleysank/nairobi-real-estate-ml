import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("C:/Users/hp/Downloads/Nairobi_prices.csv")
print(df.info())

# --- DATA CLEANING ---

import pandas as pd

# 1. Load the data
df = pd.read_csv("C:/Users/hp/Downloads/Nairobi_prices.csv")

# --- DATA CLEANING ---

# Clean Price: Using regex=True to handle multiple case variations
# We replace anything that IS NOT a digit (0-9) with an empty string
df['Price'] = (df['Price']
               .str.replace(r'[^\d]', '', regex=True) # Removes all non-numeric chars (Ksh, spaces, commas)
               .str.strip())

# Convert to float safely
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Handle Missing Values
df['Bedroom'] = df['Bedroom'].fillna(df['Bedroom'].median())
df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())

# Verification & Statistics
print("--- Cleaned Data Info ---")
print(df.info())

print("\n--- Price Statistics (Millions KES) ---")
# Dividing by 1,000,000 for readability
print((df['Price'] / 1_000_000).describe())

import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot with a regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Bedroom', y='Price', scatter_kws={'alpha':0.5})
plt.title('Relationship: Number of Bedrooms vs. House Price')
plt.ylabel('Price (KES)')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# FEATURE ENGINEERING ---
# Convert 'Location' into numeric columns (One-Hot Encoding)
# We use Bedroom, bathroom, and Location as our deterministic factors
X = pd.get_dummies(df[['Bedroom', 'bathroom', 'Location']], drop_first=True)
y = df['Price']

# DATA SPLITTING ---
# We keep 20% of the data aside to test the model's accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL TRAINING ---
model = LinearRegression()
model.fit(X_train, y_train)

# MODEL EVALUATION ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n--- Model Performance ---")
print(f"Average Error (MAE): KES {mae:,.0f}")
print(f"R-Squared Score: {r2:.2f}")

# EXTRACTING DETERMINISTIC FACTORS ---
# Create a DataFrame to view the 'Weight' of each feature
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print("\n--- Feature Impact (Marginal Price Increase) ---")
print(coefficients.loc[['Bedroom', 'bathroom']])

# Filter for the top 5 most influential locations
location_coeffs = coefficients.drop(['Bedroom', 'bathroom']).sort_values(by='Coefficient', ascending=False)
print("\n--- Top 5 Location Premiums ---")
print(location_coeffs.head(5))

# CUSTOM PREDICTION TOOL ---
def predict_my_house(beds, baths, location):
    # Create a template matching the model's structure
    sample = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # Fill in the known values
    sample['Bedroom'] = beds
    sample['bathroom'] = baths
    
    loc_col = 'Location_' + location
    if loc_col in sample.columns:
        sample[loc_col] = 1
        
    price = model.predict(sample)[0]
    return price

# Example Test: 4 Bedroom, 4 Bathroom in Runda
est_price = predict_my_house(4, 4, 'Runda')
print(f"\nEstimated Price for 4BR in Runda: KES {est_price:,.0f}")