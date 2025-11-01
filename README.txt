# Week 1 Assignment: Data Exploration, Cleaning, and Feature Engineering

##  Objective
This project demonstrates data exploration, cleaning, outlier handling, and feature engineering using Python.  

---

## Dataset
**Name:** Synthetic Online Retail Dataset  
**Rows:** 1,000  
**Columns:** 13  

### Key Columns
- `order_date`: transaction date  
- `category_name`: product category  
- `price`, `quantity`: numerical variables  
- `review_score`, `gender`, `age`: customer features  

---

##   Analysis Pipeline
1. **Exploratory Data Analysis**
   - Summary statistics, data types, missingness
   - Visualizations: distributions, boxplots, categorical breakdowns

2. **Data Cleaning**
   - Missing values handled using median/mean/mode imputation
   - Outliers detected via IQR; capped price at 95th percentile

3. **Feature Engineering**
   - `total_value` = quantity × price  
   - `purchase_month` extracted from order_date  
   - `price_vs_category_avg` (relative price)  
   - `age_group` (binned age ranges)

4. **Feature Scaling**
   - StandardScaler for approximately normal features (`age`, `review_score`)  
   - MinMaxScaler for bounded/skewed features (`price`, `quantity`, etc.)

---

## Tech Stack
- **Language:** Python 3.10+
- **IDE:** Visual Studio Code
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

---
