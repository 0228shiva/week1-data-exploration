"""
Week 1 Assignment: Data Exploration, Cleaning, and Feature Engineering
Main entry point for the analysis pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath='/content/sample_data/synthetic_online_retail_data.csv'):
    """Load the retail dataset from Excel file"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def exploratory_data_analysis(df):
    """Perform comprehensive EDA on the dataset"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Basic information
    print("\n1. Dataset Overview:")
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {df.shape[1]}")
    print(f"\n   Column Names: {list(df.columns)}")

    # Data types
    print("\n2. Data Types:")
    print(df.dtypes)

    # Statistical summary
    print("\n3. Statistical Summary:")
    print(df.describe())

    # Missing values
    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    # Unique values for categorical columns
    print("\n5. Cardinality (Unique Values):")
    categorical_cols = ['category_name', 'product_name', 'payment_method', 'city', 'gender']
    for col in categorical_cols:
        if col in df.columns:
            print(f"   {col}: {df[col].nunique()} unique values")

    # Create visualizations
    create_eda_visualizations(df)

    return df


def create_eda_visualizations(df):
    """Create EDA visualizations"""
    print("\n6. Creating Visualizations...")

    # Figure 1: Distribution of numerical variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Numerical Variables', fontsize=16, y=1.02)

    numerical_cols = ['quantity', 'price', 'review_score', 'age']
    for idx, col in enumerate(numerical_cols):
        if col in df.columns:
            row, col_idx = idx // 3, idx % 3
            axes[row, col_idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[row, col_idx].set_title(f'{col} Distribution')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')

    # Boxplots for outlier detection
    axes[1, 0].boxplot(df['price'].dropna())
    axes[1, 0].set_title('Price - Boxplot (Outlier Detection)')
    axes[1, 0].set_ylabel('Price')

    axes[1, 1].boxplot(df['quantity'].dropna())
    axes[1, 1].set_title('Quantity - Boxplot')
    axes[1, 1].set_ylabel('Quantity')

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/eda_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: outputs/eda_distributions.png")

    # Figure 2: Categorical analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Categorical Variables Analysis', fontsize=16, y=1.02)

    # Category distribution
    category_counts = df['category_name'].value_counts()
    axes[0, 0].bar(range(len(category_counts)), category_counts.values)
    axes[0, 0].set_xticks(range(len(category_counts)))
    axes[0, 0].set_xticklabels(category_counts.index, rotation=45, ha='right')
    axes[0, 0].set_title('Distribution by Category')
    axes[0, 0].set_ylabel('Count')

    # Payment method
    payment_counts = df['payment_method'].value_counts()
    axes[0, 1].bar(range(len(payment_counts)), payment_counts.values)
    axes[0, 1].set_xticks(range(len(payment_counts)))
    axes[0, 1].set_xticklabels(payment_counts.index, rotation=45, ha='right')
    axes[0, 1].set_title('Payment Methods')
    axes[0, 1].set_ylabel('Count')

    # Gender distribution
    gender_counts = df['gender'].value_counts()
    axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Gender Distribution')

    # Review score distribution
    review_counts = df['review_score'].value_counts().sort_index()
    axes[1, 1].bar(review_counts.index, review_counts.values)
    axes[1, 1].set_title('Review Score Distribution')
    axes[1, 1].set_xlabel('Review Score')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('outputs/eda_categorical.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: outputs/eda_categorical.png")
    plt.close('all')


def detect_outliers(df):
    """Detect outliers using IQR method and visualizations"""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)

    outlier_info = {}
    numerical_cols = ['quantity', 'price', 'age']

    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

            print(f"\n{col.upper()}:")
            print(f"   IQR Method: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
            print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"   Outliers: {len(outliers)} ({outlier_info[col]['percentage']:.2f}%)")

    return outlier_info


def clean_data(df):
    """Clean the dataset: handle missing values and outliers"""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    df_clean = df.copy()

    # Handle missing values
    print("\n1. Handling Missing Values:")

    # Review score: Impute with median (MCAR assumption - missing at random)
    if df_clean['review_score'].isnull().sum() > 0:
        median_score = df_clean['review_score'].median()
        df_clean['review_score'].fillna(median_score, inplace=True)
        print(f"   ✓ review_score: Filled {df['review_score'].isnull().sum()} missing values with median ({median_score})")

    # Age: Impute with mean (MAR assumption)
    if df_clean['age'].isnull().sum() > 0:
        mean_age = df_clean['age'].mean()
        df_clean['age'].fillna(mean_age, inplace=True)
        print(f"   ✓ age: Filled {df['age'].isnull().sum()} missing values with mean ({mean_age:.1f})")

    # Gender: Impute with mode
    if df_clean['gender'].isnull().sum() > 0:
        mode_gender = df_clean['gender'].mode()[0]
        df_clean['gender'].fillna(mode_gender, inplace=True)
        print(f"   ✓ gender: Filled {df['gender'].isnull().sum()} missing values with mode ({mode_gender})")

    # Handle outliers
    print("\n2. Handling Outliers:")

    # Price outliers: Cap at 95th percentile (business decision - keeps data)
    price_95 = df_clean['price'].quantile(0.95)
    outliers_before = len(df_clean[df_clean['price'] > price_95])
    df_clean['price'] = df_clean['price'].clip(upper=price_95)
    print(f"   ✓ price: Capped {outliers_before} values at 95th percentile ({price_95:.2f})")

    # Quantity outliers: Keep (valid business scenario - bulk orders)
    print(f"   ✓ quantity: Retained all values (bulk orders are valid)")

    # Verify no missing values remain
    assert df_clean.isnull().sum().sum() == 0, "Missing values still present!"
    print("\n   ✓ All missing values handled successfully")

    return df_clean


def feature_engineering(df):
    """Engineer new features for downstream modeling"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df_features = df.copy()

    # Feature 1: Total Transaction Value
    df_features['total_value'] = df_features['quantity'] * df_features['price']
    print("\n1. Created 'total_value' = quantity × price")
    print(f"   Rationale: Captures transaction magnitude for customer segmentation")
    print(f"   Range: ${df_features['total_value'].min():.2f} to ${df_features['total_value'].max():.2f}")
    print(f"   Mean: ${df_features['total_value'].mean():.2f}")

    # Feature 2: Purchase Month
    df_features['order_date'] = pd.to_datetime(df_features['order_date'])
    df_features['purchase_month'] = df_features['order_date'].dt.month
    print("\n2. Created 'purchase_month' (extracted from order_date)")
    print(f"   Rationale: Captures seasonality patterns in purchasing behavior")
    print(f"   Distribution: {df_features['purchase_month'].value_counts().sort_index().to_dict()}")

    # Feature 3: Price per Unit Category (normalized price)
    category_avg_price = df_features.groupby('category_name')['price'].transform('mean')
    df_features['price_vs_category_avg'] = df_features['price'] / category_avg_price
    print("\n3. Created 'price_vs_category_avg' = price / category_average_price")
    print(f"   Rationale: Indicates premium vs budget items within categories")
    print(f"   Interpretation: >1 = above average, <1 = below average")
    print(f"   Mean: {df_features['price_vs_category_avg'].mean():.2f}")

    # Feature 4: Customer Age Group
    df_features['age_group'] = pd.cut(df_features['age'],
                                       bins=[0, 25, 35, 50, 100],
                                       labels=['18-25', '26-35', '36-50', '50+'])
    print("\n4. Created 'age_group' (binned age)")
    print(f"   Rationale: Simplifies age for demographic segmentation")
    print(f"   Distribution:\n{df_features['age_group'].value_counts()}")

    return df_features


def apply_scaling(df):
    """Apply appropriate scaling to numerical features"""
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    df_scaled = df.copy()

    # StandardScaler for features with normal-ish distributions
    standard_cols = ['age', 'review_score']
    scaler_standard = StandardScaler()
    df_scaled[standard_cols] = scaler_standard.fit_transform(df[standard_cols])

    print("\n1. StandardScaler applied to:", standard_cols)
    print("   Rationale: These features have approximately normal distributions")
    print("   Result: Mean ≈ 0, Std ≈ 1")
    for col in standard_cols:
        print(f"   {col}: mean={df_scaled[col].mean():.4f}, std={df_scaled[col].std():.4f}")

    # MinMaxScaler for features with bounded ranges or skewed distributions
    minmax_cols = ['quantity', 'price', 'total_value', 'price_vs_category_avg']
    scaler_minmax = MinMaxScaler()
    df_scaled[minmax_cols] = scaler_minmax.fit_transform(df[minmax_cols])

    print("\n2. MinMaxScaler applied to:", minmax_cols)
    print("   Rationale: Preserves relationships, bounded to [0,1] for stability")
    print("   Result: Range = [0, 1]")
    for col in minmax_cols:
        print(f"   {col}: min={df_scaled[col].min():.4f}, max={df_scaled[col].max():.4f}")

    # Save scaling comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Before vs After Scaling', fontsize=16)

    cols_to_plot = ['age', 'price', 'total_value']
    for idx, col in enumerate(cols_to_plot):
        # Before
        axes[0, idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[0, idx].set_title(f'{col} (Original)')
        axes[0, idx].set_ylabel('Frequency')

        # After
        axes[1, idx].hist(df_scaled[col], bins=30, edgecolor='black', alpha=0.7)
        axes[1, idx].set_title(f'{col} (Scaled)')
        axes[1, idx].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('outputs/scaling_comparison.png', dpi=300, bbox_inches='tight')
    print("\n   ✓ Saved: outputs/scaling_comparison.png")
    plt.close()

    return df_scaled


def save_results(df_original, df_cleaned, df_engineered, df_scaled):
    """Save processed datasets"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    df_cleaned.to_csv('outputs/data_cleaned.csv', index=False)
    print("✓ Saved: outputs/data_cleaned.csv")

    df_engineered.to_csv('outputs/data_with_features.csv', index=False)
    print("✓ Saved: outputs/data_with_features.csv")

    df_scaled.to_csv('outputs/data_scaled.csv', index=False)
    print("✓ Saved: outputs/data_scaled.csv")

    # Summary statistics
    summary = {
        'Original Rows': len(df_original),
        'Final Rows': len(df_scaled),
        'Original Columns': len(df_original.columns),
        'Final Columns': len(df_scaled.columns),
        'New Features': len(df_scaled.columns) - len(df_original.columns)
    }

    print("\nPipeline Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")


def main():
    """Main execution pipeline"""
    print("\n" + "=" * 60)
    print("WEEK 1 ASSIGNMENT: DATA ANALYSIS PIPELINE")
    print("Synthetic Online Retail Data")
    print("=" * 60 + "\n")

    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)

    # 1. Load data
    df = load_data('/content/sample_data/synthetic_online_retail_data.csv')

    # 2. EDA
    df = exploratory_data_analysis(df)

    # 3. Detect outliers
    outlier_info = detect_outliers(df)

    # 4. Clean data
    df_cleaned = clean_data(df)

    # 5. Feature engineering
    df_engineered = feature_engineering(df_cleaned)

    # 6. Apply scaling
    df_scaled = apply_scaling(df_engineered)

    # 7. Save results
    save_results(df, df_cleaned, df_engineered, df_scaled)

    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

