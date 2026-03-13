# Nairobi House Price Prediction Model 
 
## Project Overview 
This project uses Multiple Linear Regression to analyze and predict property prices in Nairobi. 
 
## Economic Interpretation 
* **R-Squared (0.71):** Approximately **71% of the variation** in Nairobi house prices is deterministic, explained by the number of **bedrooms, bathrooms, and location**. 
* **Key Determinants:** Location proved to be a major factor, with areas like Rosslyn and Runda commanding significant "Location Premiums." 
* **Marginal Effects:** According to the model coefficients, an additional bedroom adds approximately KES 28M to the valuation, holding all other factors constant. 
 
## Technical Implementation 
* **Cleaning:** Robust regex cleaning to handle currency strings and space-separated thousands. 
* **Preprocessing:** One-Hot Encoding for categorical neighborhood data and median imputation for missing values. 
 
## Visualizing Accuracy 
![Model Accuracy Plot](model_accuracy_plot.png) 
