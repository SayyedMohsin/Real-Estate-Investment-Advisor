"""
Real Estate Investment Advisor - Complete Training in One File
Run this file once to train models: python train_model.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, r2_score)
import joblib
import os

print("=" * 60)
print("🏠 REAL ESTATE INVESTMENT ADVISOR - MODEL TRAINING")
print("=" * 60)

def load_and_prepare_data():
    """Load and prepare the data"""
    print("\n📊 Loading data...")
    
    try:
        # Try to load the data
        df = pd.read_csv('real_estate_data.csv')
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info
        print(f"\n📈 Basic Statistics:")
        print(f"   Average Price: ₹{df['Price_in_Lakhs'].mean():.2f} L")
        print(f"   Price Range: ₹{df['Price_in_Lakhs'].min():.2f} - ₹{df['Price_in_Lakhs'].max():.2f} L")
        print(f"   Average Size: {df['Size_in_SqFt'].mean():.0f} sq ft")
        print(f"   Most Common City: {df['City'].mode()[0]}")
        print(f"   Most Common Property Type: {df['Property_Type'].mode()[0]}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: 'real_estate_data.csv' not found!")
        print("\n📥 Please download the dataset and save it as 'real_estate_data.csv' in this folder.")
        print("   Dataset URL: https://drive.google.com/file/d/1OySoqcM7IAr6q9UBRbGYR-BV2dvee_Sk/view?usp=sharing")
        return None

def clean_data(df):
    """Clean the data properly"""
    print("\n🧹 Cleaning data...")
    
    # Make a copy
    data = df.copy()
    
    # Display initial info
    print(f"   Initial records: {len(data)}")
    
    # Step 1: Handle missing values in key columns
    print("\n   Handling missing values...")
    
    # List of columns to check
    important_columns = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'City', 'Property_Type']
    
    for col in important_columns:
        if col in data.columns:
            missing = data[col].isnull().sum()
            if missing > 0:
                print(f"      {col}: {missing} missing values")
    
    # Remove rows with missing critical values
    initial_rows = len(data)
    data = data.dropna(subset=['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'City'])
    removed = initial_rows - len(data)
    print(f"   Removed {removed} records with missing critical data")
    
    # Step 2: Remove unrealistic values
    initial_rows = len(data)
    
    # Remove negative or zero prices
    data = data[data['Price_in_Lakhs'] > 0]
    
    # Remove unrealistic sizes (too small or too large)
    data = data[(data['Size_in_SqFt'] > 100) & (data['Size_in_SqFt'] < 10000)]
    
    # Remove unrealistic BHK
    data = data[(data['BHK'] >= 1) & (data['BHK'] <= 10)]
    
    # Remove unrealistic years
    if 'Year_Built' in data.columns:
        current_year = 2024
        data = data[(data['Year_Built'] >= 1900) & (data['Year_Built'] <= current_year)]
    
    removed = initial_rows - len(data)
    print(f"   Removed {removed} unrealistic records")
    print(f"   Remaining: {len(data)} records")
    
    # Step 3: Convert columns to proper types
    print("\n   Converting columns...")
    
    numeric_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Year_Built',
                   'Floor_No', 'Total_Floors', 'Nearby_Schools', 
                   'Nearby_Hospitals', 'Parking_Space']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill with median
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    
    # Handle Public_Transport_Accessibility
    if 'Public_Transport_Accessibility' in data.columns:
        # Extract numeric values from string
        data['Public_Transport_Accessibility'] = (
            data['Public_Transport_Accessibility']
            .astype(str)
            .str.extract('(\d+)')[0]
        )
        data['Public_Transport_Accessibility'] = pd.to_numeric(
            data['Public_Transport_Accessibility'], errors='coerce'
        )
        # Fill missing with median
        median_val = data['Public_Transport_Accessibility'].median()
        data['Public_Transport_Accessibility'].fillna(median_val, inplace=True)
    
    # Fill other numeric columns with median
    numeric_cols_all = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols_all:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().any():
            mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
            data[col].fillna(mode_val, inplace=True)
    
    # Step 4: Create derived columns
    print("\n   Creating derived columns...")
    
    # Create Price_per_SqFt
    data['Price_per_SqFt'] = (data['Price_in_Lakhs'] * 100000) / data['Size_in_SqFt']
    
    # Remove infinite values
    data['Price_per_SqFt'].replace([np.inf, -np.inf], np.nan, inplace=True)
    data['Price_per_SqFt'].fillna(data['Price_per_SqFt'].median(), inplace=True)
    
    # Create Age_of_Property
    current_year = 2024
    if 'Year_Built' in data.columns:
        data['Age_of_Property'] = current_year - data['Year_Built']
        data['Age_of_Property'] = data['Age_of_Property'].clip(0, 100)
    else:
        # Create random age if Year_Built not available
        data['Age_of_Property'] = np.random.randint(0, 30, len(data))
    
    print("✅ Data cleaning completed")
    
    # Display final statistics
    print(f"\n📊 Cleaned Data Statistics:")
    print(f"   Total properties: {len(data):,}")
    print(f"   Average price: ₹{data['Price_in_Lakhs'].mean():.2f} L")
    print(f"   Average size: {data['Size_in_SqFt'].mean():.0f} sq ft")
    print(f"   Average price per sq ft: ₹{data['Price_per_SqFt'].mean():.0f}")
    
    return data

def create_features_and_targets(data):
    """Create features and target variables"""
    print("\n🔧 Creating features and targets...")
    
    # Select features for modeling (simplified)
    features = [
        'BHK', 
        'Size_in_SqFt', 
        'Price_per_SqFt', 
        'Age_of_Property',
        'Nearby_Schools', 
        'Nearby_Hospitals',
        'Public_Transport_Accessibility', 
        'Parking_Space',
        'Property_Type', 
        'City'
    ]
    
    # Keep only features that exist
    available_features = []
    for f in features:
        if f in data.columns:
            available_features.append(f)
        else:
            print(f"   Warning: {f} not found in data")
    
    print(f"   Using {len(available_features)} features")
    
    # Create feature matrix
    X = data[available_features].copy()
    
    # Check for NaN in features
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"   Warning: {nan_count} NaN values in features, filling with median/mode")
        
        # Fill numeric NaNs with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        # Fill categorical NaNs with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X[col].fillna(mode_val, inplace=True)
    
    # Create targets
    
    # 1. REGRESSION TARGET: Future price after 5 years
    print("\n   Creating regression target...")
    
    # Base appreciation rate
    base_appreciation = 0.08  # 8% per year
    
    # Calculate base future price
    future_price = data['Price_in_Lakhs'] * ((1 + base_appreciation) ** 5)
    
    # Add city-specific variation
    if 'City' in data.columns:
        for city in data['City'].unique():
            city_mask = data['City'] == city
            city_count = sum(city_mask)
            
            if city_count > 10:
                # Calculate city statistics
                city_prices = data.loc[city_mask, 'Price_in_Lakhs']
                city_mean = city_prices.mean()
                city_std = city_prices.std()
                
                if city_std > 0:
                    # Add random variation based on city volatility
                    variation = np.random.normal(0, city_std * 0.05, city_count)
                    future_price.loc[city_mask] = future_price.loc[city_mask] * (1 + variation / city_mean)
    
    y_regression = future_price
    
    # Ensure no negative prices
    y_regression = y_regression.clip(lower=data['Price_in_Lakhs'] * 0.5, 
                                    upper=data['Price_in_Lakhs'] * 3)
    
    print(f"   Future price range: ₹{y_regression.min():.2f} - ₹{y_regression.max():.2f} L")
    
    # 2. CLASSIFICATION TARGET: Good Investment
    print("\n   Creating classification target...")
    
    # Simple scoring system
    investment_score = np.zeros(len(data))
    
    # Rule 1: Price below city median (2 points)
    if 'City' in data.columns:
        city_medians = data.groupby('City')['Price_in_Lakhs'].transform('median')
        investment_score += (data['Price_in_Lakhs'] <= city_medians).astype(int) * 2
    
    # Rule 2: Price per SqFt below city median (2 points)
    if 'City' in data.columns:
        pps_medians = data.groupby('City')['Price_per_SqFt'].transform('median')
        investment_score += (data['Price_per_SqFt'] <= pps_medians).astype(int) * 2
    
    # Rule 3: Good BHK (2-4 is optimal) (1 point)
    investment_score += ((data['BHK'] >= 2) & (data['BHK'] <= 4)).astype(int)
    
    # Rule 4: Newer property (<= 10 years) (1 point)
    investment_score += (data['Age_of_Property'] <= 10).astype(int)
    
    # Rule 5: Good amenities (1 point)
    if 'Nearby_Schools' in data.columns and 'Nearby_Hospitals' in data.columns:
        good_amenities = (data['Nearby_Schools'] >= 5) & (data['Nearby_Hospitals'] >= 5)
        investment_score += good_amenities.astype(int)
    
    # Create binary classification (score >= 4 is good investment)
    # Max possible score is 7, so threshold at 4
    y_classification = (investment_score >= 4).astype(int)
    
    good_count = y_classification.sum()
    good_percentage = (good_count / len(y_classification)) * 100
    
    print(f"   Good investments: {good_count:,} out of {len(y_classification):,} ({good_percentage:.1f}%)")
    
    return X, y_regression, y_classification, available_features

def train_models(X, y_class, y_reg, feature_names):
    """Train classification and regression models with proper NaN handling"""
    print("\n🤖 Training models...")
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"   Numerical features: {len(numerical_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    # Create preprocessing pipeline with imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaN
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle NaN
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Split data
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Check for NaN after split
    print(f"   NaN in X_train: {X_train.isnull().sum().sum()}")
    print(f"   NaN in X_test: {X_test.isnull().sum().sum()}")
    
    # Train Classification Model
    print("\n   🎯 Training Classification Model...")
    
    # Use simpler model if data is large
    n_estimators = 50 if len(X_train) > 100000 else 100
    
    class_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    class_pipeline.fit(X_train, y_train_class)
    
    # Evaluate classification
    y_pred_class = class_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, zero_division=0)
    recall = recall_score(y_test_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_test_class, y_pred_class, zero_division=0)
    
    print(f"   ✅ Classification Results:")
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1 Score:  {f1:.4f}")
    
    # Train Regression Model
    print("\n   💰 Training Regression Model...")
    
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    reg_pipeline.fit(X_train, y_train_reg)
    
    # Evaluate regression
    y_pred_reg = reg_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    # Calculate MAPE (handle zero values)
    y_test_nonzero = y_test_reg[y_test_reg != 0]
    y_pred_nonzero = y_pred_reg[y_test_reg != 0]
    
    if len(y_test_nonzero) > 0:
        mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
    else:
        mape = 0
    
    print(f"   ✅ Regression Results:")
    print(f"      RMSE: ₹{rmse:.2f} L")
    print(f"      R² Score: {r2:.4f}")
    print(f"      MAPE: {mape:.2f}%")
    
    # Save everything
    print("\n💾 Saving models and information...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(class_pipeline, 'models/classification_model.pkl')
    joblib.dump(reg_pipeline, 'models/regression_model.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save preprocessing info
    preprocessing_info = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'all_features': feature_names
    }
    joblib.dump(preprocessing_info, 'models/preprocessing_info.pkl')
    
    print("✅ Models saved in 'models/' directory:")
    print("   - classification_model.pkl")
    print("   - regression_model.pkl")
    print("   - feature_names.pkl")
    print("   - preprocessing_info.pkl")
    
    return class_pipeline, reg_pipeline, accuracy, r2

def main():
    """Main training function"""
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Clean data
    cleaned_data = clean_data(df)
    
    # Sample the data if it's too large (for faster training)
    if len(cleaned_data) > 100000:
        print(f"\n⚠️ Large dataset detected ({len(cleaned_data):,} records)")
        print("   Sampling 100,000 records for faster training...")
        cleaned_data = cleaned_data.sample(n=100000, random_state=42)
        print(f"   Using {len(cleaned_data):,} records for training")
    
    # Create features and targets
    X, y_reg, y_class, feature_names = create_features_and_targets(cleaned_data)
    
    # Train models
    class_model, reg_model, accuracy, r2 = train_models(X, y_class, y_reg, feature_names)
    
    # Create sample prediction
    print("\n🎯 SAMPLE PREDICTION:")
    print("-" * 40)
    
    # Take a sample property
    sample_idx = np.random.randint(0, len(X))
    sample_property = X.iloc[[sample_idx]]
    
    # Make predictions
    is_good_investment = class_model.predict(sample_property)[0]
    future_price = reg_model.predict(sample_property)[0]
    
    # Get actual current price
    actual_data_idx = cleaned_data.index[sample_idx]
    current_price = cleaned_data.loc[actual_data_idx, 'Price_in_Lakhs']
    
    print(f"Property Details:")
    print(f"  City: {cleaned_data.loc[actual_data_idx, 'City']}")
    print(f"  Property Type: {cleaned_data.loc[actual_data_idx, 'Property_Type']}")
    print(f"  BHK: {cleaned_data.loc[actual_data_idx, 'BHK']}")
    print(f"  Size: {cleaned_data.loc[actual_data_idx, 'Size_in_SqFt']} sq ft")
    print(f"  Age: {cleaned_data.loc[actual_data_idx, 'Age_of_Property']} years")
    print(f"  Current Price: ₹{current_price:.2f} L")
    print(f"\nPredictions:")
    print(f"  Good Investment: {'✅ YES' if is_good_investment == 1 else '❌ NO'}")
    print(f"  Predicted Price (5 years): ₹{future_price:.2f} L")
    
    if current_price > 0:
        appreciation = ((future_price / current_price) ** (1/5) - 1) * 100
        print(f"  Expected Annual Appreciation: {appreciation:.1f}%")
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n🚀 NEXT STEP:")
    print("   Run the application: streamlit run app.py")
    print("\n📊 Model Performance Summary:")
    print(f"   Classification Accuracy: {accuracy:.2%}")
    print(f"   Regression R² Score: {r2:.2%}")
    
    # Save cleaned data for reference
    cleaned_data.to_csv('cleaned_real_estate_data.csv', index=False)
    print(f"\n📁 Cleaned data saved: cleaned_real_estate_data.csv")
    
    # Show feature importance
    try:
        print("\n🎯 Top 10 Important Features for Classification:")
        classifier = class_model.named_steps['classifier']
        feature_importance = classifier.feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = class_model.named_steps['preprocessor']
        
        # For numerical features
        num_features = numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # For categorical features (after one-hot encoding)
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Simple feature importance display
        if len(feature_importance) > 0:
            top_n = min(10, len(feature_importance))
            indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            print("   Feature importance ranking:")
            for i, idx in enumerate(indices):
                if idx < len(num_features):
                    feat_name = num_features[idx]
                else:
                    feat_name = f"Categorical_Feature_{idx - len(num_features)}"
                print(f"   {i+1}. {feat_name}: {feature_importance[idx]:.4f}")
    except:
        print("   Feature importance analysis skipped")

if __name__ == "__main__":
    main()