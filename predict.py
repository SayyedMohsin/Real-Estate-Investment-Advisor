"""
Prediction Script - Test your models
Run: python predict.py
"""

import joblib
import pandas as pd
import numpy as np

def predict_sample():
    """Make predictions on a sample property"""
    print("=" * 60)
    print("🏠 REAL ESTATE INVESTMENT ADVISOR - PREDICTION TEST")
    print("=" * 60)
    
    try:
        # Load models
        print("\n📂 Loading models...")
        class_model = joblib.load('models/classification_model.pkl')
        reg_model = joblib.load('models/regression_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        print("✅ Models loaded successfully")
        
        # Create a sample property
        print("\n🎯 Creating sample property for prediction...")
        
        sample_property = {
            'BHK': 3,
            'Size_in_SqFt': 1500,
            'Price_per_SqFt': 8000,  # This will be calculated
            'Age_of_Property': 5,
            'Floor_No': 2,
            'Total_Floors': 10,
            'Nearby_Schools': 7,
            'Nearby_Hospitals': 6,
            'Public_Transport_Accessibility': 8,
            'Parking_Space': 2,
            'Property_Type': 'Apartment',
            'Furnished_Status': 'Semi-Furnished',
            'City': 'Mumbai',
            'Facing': 'North'
        }
        
        # Calculate Price_per_SqFt from Price_in_Lakhs
        price_in_lakhs = 120  # 1.2 Crore
        sample_property['Price_per_SqFt'] = (price_in_lakhs * 100000) / sample_property['Size_in_SqFt']
        
        # Create DataFrame
        X_sample = pd.DataFrame([sample_property])
        
        # Ensure all feature columns are present
        for col in feature_names:
            if col not in X_sample.columns:
                X_sample[col] = 0
        
        # Reorder columns
        X_sample = X_sample[feature_names]
        
        # Make predictions
        print("\n🔮 Making predictions...")
        is_good_investment = class_model.predict(X_sample)[0]
        future_price = reg_model.predict(X_sample)[0]
        
        # Calculate appreciation
        appreciation = ((future_price / price_in_lakhs) ** (1/5) - 1) * 100
        
        # Display results
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 40)
        print(f"Property Details:")
        print(f"  Location: {sample_property['City']}")
        print(f"  Type: {sample_property['Property_Type']} ({sample_property['BHK']} BHK)")
        print(f"  Size: {sample_property['Size_in_SqFt']} sq ft")
        print(f"  Age: {sample_property['Age_of_Property']} years")
        print(f"  Current Price: ₹{price_in_lakhs:.2f} L")
        print(f"\nPredictions:")
        print(f"  Investment Potential: {'✅ GOOD INVESTMENT' if is_good_investment == 1 else '⚠️ RECONSIDER'}")
        print(f"  Predicted Price (5 years): ₹{future_price:.2f} L")
        print(f"  Expected Annual Appreciation: {appreciation:.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ PREDICTION TEST COMPLETED")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n⚠️  Models not found! Please run training first:")
        print("   python train_model.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    predict_sample()