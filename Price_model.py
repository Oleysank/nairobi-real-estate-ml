import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD DATA
df = pd.read_csv("C:/Users/hp/Downloads/Nairobi_prices.csv")

# 2. DATA CLEANING
# Clean Price
df['Price'] = (df['Price']
               .str.replace(r'[^\d]', '', regex=True)
               .replace('', np.nan)
               .astype(float))

# Clean House Size (New Feature!)
df['House size'] = (df['House size']
                    .str.replace(r'[^\d.]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float))

# Fill Missing Values with Medians
df['Bedroom'] = df['Bedroom'].fillna(df['Bedroom'].median())
df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())
df['House size'] = df['House size'].fillna(df['House size'].median())

# 3. FEATURE ENGINEERING & LOG TRANSFORM
# We use Bedroom, bathroom, and House size as numerical factors
X = pd.get_dummies(df[['Bedroom', 'bathroom', 'House size', 'Location']], drop_first=True)
y = np.log(df['Price']) # Log transform to handle market skewness

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

# 6. EVALUATION
predictions_log = model.predict(X_test)
mae = mean_absolute_error(np.exp(y_test), np.exp(predictions_log))
r2 = r2_score(y_test, predictions_log)

print(f"\n--- Model Performance ---")
print(f"Average Error (MAE): KES {mae:,.0f}")
print(f"R-Squared Score: {r2:.2f}")

# 7. PREDICTION FUNCTION
def predict_house(beds, baths, size, location):
    # Create an empty row with all columns
    sample = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # Fill numerical values
    sample['Bedroom'] = beds
    sample['bathroom'] = baths
    sample['House size'] = size
    
    # Set location to 1
    loc_col = f'Location_{location}'
    if loc_col in sample.columns:
        sample[loc_col] = 1
        
    log_price = model.predict(sample)[0]
    return np.exp(log_price) # Return real KES

# 8. INTERACTIVE LOOP
print("\n" + "="*35)
print(" NAIROBI REAL ESTATE VALUATOR ")
print("="*35)

while True:
    loc = input("\nEnter Location (or 'exit'): ").title()
    if loc.lower() == 'exit': break
    
    try:
        beds = float(input("Bedrooms: "))
        baths = float(input("Bathrooms: "))
        size = float(input("House Size (sqft/sqm): "))
        
        final_val = predict_house(beds, baths, size, loc)
        print(f"\n>>> Estimated Market Value: KES {final_val:,.0f}")
    except Exception as e:
        print(f"Error: {e}. Check spelling and try again.")