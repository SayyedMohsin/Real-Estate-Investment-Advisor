#!/usr/bin/env python3
"""
Real Estate Investment Advisor - Main Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_pipeline(data_path):
    """
    Run the complete ML pipeline
    """
    print("=" * 60)
    print("🏠 REAL ESTATE INVESTMENT ADVISOR - ML PIPELINE")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    try:
        # ============================================
        # STEP 1: DATA PREPROCESSING
        # ============================================
        print("\n📊 STEP 1: DATA PREPROCESSING")
        print("-" * 40)
        
        preprocessor = DataPreprocessor(data_path)
        
        # Load and display data info
        data = preprocessor.load_data()
        
        # Clean the data
        cleaned_data = preprocessor.clean_data()
        
        # Save cleaned data
        cleaned_data_path = 'data/processed/cleaned_data.csv'
        preprocessor.save_cleaned_data(cleaned_data_path)
        
        # Display basic statistics
        print("\n📈 CLEANED DATA STATISTICS:")
        print(f"   Total properties: {len(cleaned_data):,}")
        print(f"   Average price: ₹{cleaned_data['Price_in_Lakhs'].mean():.2f} L")
        print(f"   Price range: ₹{cleaned_data['Price_in_Lakhs'].min():.2f} - ₹{cleaned_data['Price_in_Lakhs'].max():.2f} L")
        print(f"   Average size: {cleaned_data['Size_in_SqFt'].mean():.0f} sq ft")
        print(f"   Most common property type: {cleaned_data['Property_Type'].mode()[0]}")
        
        # ============================================
        # STEP 2: FEATURE ENGINEERING
        # ============================================
        print("\n\n🔧 STEP 2: FEATURE ENGINEERING")
        print("-" * 40)
        
        engineer = FeatureEngineer(cleaned_data)
        
        # Prepare features
        X, y_reg, y_class = engineer.prepare_features()
        
        # Save features data
        features_path = 'data/processed/features_data.csv'
        engineer.save_features(features_path)
        
        # Display feature information
        print("\n📊 FEATURE INFORMATION:")
        print(f"   Total features: {len(X.columns)}")
        print(f"   Numerical features: {len(X.select_dtypes(include=[np.number]).columns)}")
        print(f"   Categorical features: {len(X.select_dtypes(include=['object']).columns)}")
        
        print(f"\n💰 TARGET VARIABLES:")
        print(f"   Future Price (5Y): Mean = ₹{y_reg.mean():.2f} L, Range = ₹{y_reg.min():.2f} - ₹{y_reg.max():.2f} L")
        print(f"   Good Investment: {y_class.sum():,} good investments out of {len(y_class):,} ({(y_class.mean()*100):.1f}%)")
        
        # ============================================
        # STEP 3: MODEL TRAINING
        # ============================================
        print("\n\n🤖 STEP 3: MODEL TRAINING")
        print("-" * 40)
        
        # Initialize model trainer
        trainer = ModelTrainer(X, y_class, y_reg)
        
        # Run complete training pipeline
        success = trainer.run_training_pipeline()
        
        if not success:
            print("\n❌ Training pipeline failed!")
            return False
        
        # ============================================
        # FINAL SUMMARY
        # ============================================
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📁 FILES GENERATED:")
        print("-" * 40)
        print(f"✅ data/processed/cleaned_data.csv")
        print(f"✅ data/processed/features_data.csv")
        print(f"✅ models/classification_random_forest.pkl")
        print(f"✅ models/regression_random_forest.pkl")
        print(f"✅ models/feature_columns.pkl")
        print(f"✅ models/preprocessing_info.pkl")
        print(f"✅ mlruns/ (MLflow experiment tracking)")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 40)
        print("1. 📊 Start the application:")
        print("   streamlit run app_simple.py")
        print()
        print("2. 📈 View MLflow experiments (in new terminal):")
        print("   mlflow ui")
        print("   Then open: http://localhost:5000")
        print()
        print("3. 🔍 Check models in 'models/' directory")
        print()
        print("4. 📋 Test predictions:")
        print("   python test_prediction.py")
        
        # Create a test prediction script
        create_test_script()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_test_script():
    """Create a test prediction script"""
    test_script = """#!/usr/bin/env python3
"""