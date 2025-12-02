#!/usr/bin/env python3
"""
Simple script to run the training pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import after adding path
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

def simple_training():
    """Simple training pipeline without MLflow"""
    print("=" * 60)
    print("SIMPLE TRAINING PIPELINE")
    print("=" * 60)
    
    # Check if data exists
    data_path = 'data/raw/real_estate_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Data not found at: {data_path}")
        print("Please place your dataset at: data/raw/real_estate_data.csv")
        return False
    
    try:
        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Step 1: Load and clean data
        print("\n1. 📊 Loading and cleaning data...")
        preprocessor = DataPreprocessor(data_path)
        cleaned_data = preprocessor.clean_data()
        
        # Save cleaned data
        cleaned_data.to_csv('data/processed/cleaned_data.csv', index=False)
        print(f"   ✅ Cleaned data saved: {len(cleaned_data)} records")
        
        # Step 2: Feature engineering
        print("\n2. 🔧 Creating features...")
        engineer = FeatureEngineer(cleaned_data)
        X, y_reg, y_class = engineer.prepare_features()
        
        # Save features
        features_df = pd.concat([X, y_reg.rename('Future_Price_5Y'), 
                               y_class.rename('Good_Investment')], axis=1)
        features_df.to_csv('data/processed/features_data.csv', index=False)
        print(f"   ✅ Features created: {X.shape[1]} features")
        
        # Step 3: Train simple models
        print("\n3. 🤖 Training models...")
        
        # Simple train-test split
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import joblib
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        
        # Split data
        X_train, X_test, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        _, _, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Train classification model
        print("\n   Training classification model...")
        class_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        class_pipeline.fit(X_train, y_train_class)
        class_score = class_pipeline.score(X_test, y_test_class)
        print(f"   ✅ Classification accuracy: {class_score:.4f}")
        
        # Train regression model
        print("\n   Training regression model...")
        reg_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        reg_pipeline.fit(X_train, y_train_reg)
        reg_score = reg_pipeline.score(X_test, y_test_reg)
        print(f"   ✅ Regression R² score: {reg_score:.4f}")
        
        # Save models
        joblib.dump(class_pipeline, 'models/simple_classifier.pkl')
        joblib.dump(reg_pipeline, 'models/simple_regressor.pkl')
        joblib.dump(list(X.columns), 'models/simple_feature_columns.pkl')
        
        # Save preprocessing info
        preprocessing_info = {
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }
        joblib.dump(preprocessing_info, 'models/simple_preprocessing_info.pkl')
        
        print(f"\n✅ MODELS SAVED in 'models/' directory:")
        print(f"   - simple_classifier.pkl")
        print(f"   - simple_regressor.pkl")
        print(f"   - simple_feature_columns.pkl")
        print(f"   - simple_preprocessing_info.pkl")
        
        # Create sample prediction
        print("\n📊 SAMPLE PREDICTION:")
        sample_idx = 0
        sample_features = X.iloc[[sample_idx]]
        
        class_pred = class_pipeline.predict(sample_features)[0]
        reg_pred = reg_pipeline.predict(sample_features)[0]
        actual_price = y_reg.iloc[sample_idx]
        
        print(f"   Property features: {X.shape[1]} features")
        print(f"   Good investment prediction: {'YES' if class_pred == 1 else 'NO'}")
        print(f"   Current price: ₹{actual_price/(1.08**5):.2f} L")
        print(f"   Predicted future price (5Y): ₹{reg_pred:.2f} L")
        print(f"   Expected appreciation: {(reg_pred/actual_price*(1.08**5)-1)*100:.1f}%")
        
        print("\n" + "=" * 60)
        print("🎉 SIMPLE PIPELINE COMPLETED!")
        print("=" * 60)
        
        print("\n🚀 Next: Run the app with: streamlit run app_simple.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_training()