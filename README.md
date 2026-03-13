## Final Model: Multi-Feature Log-Linear Regression 
 
### Features Included: 
* **Numerical:** Bedrooms, Bathrooms, and House Size (cleaned via Regex). 
* **Categorical:** One-Hot Encoded Locations. 
 
### Methodology: 
* **Logarithmic Scaling:** Prices were log-transformed to stabilize variance across the high-value Nairobi market. 
* **Imputation:** Median values were used to handle missing data in feature columns, ensuring 100% data utilization. 
* **Deployment:** Built a terminal-based interactive tool for real-time property valuation. 
