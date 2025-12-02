import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        print(f"Feature engineering initialized on {len(self.data)} records")
        
    def prepare_features(self):
        """Main function to prepare all features"""
        print("\nPreparing features...")
        
        # Step 1: Create basic features
        self._create_basic_features()
        
        # Step 2: Create investment features (simplified)
        self._create_simple_investment_features()
        
        # Step 3: Create target variables
        self._create_target_variables()
        
        # Step 4: Select final features
        X, y_reg, y_class = self._select_final_features()
        
        print(f"\n✅ Features prepared:")
        print(f"   Features shape: {X.shape}")
        print(f"   Features: {list(X.columns)}")
        print(f"   Future price target shape: {y_reg.shape}")
        print(f"   Good investment target shape: {y_class.shape}")
        
        return X, y_reg, y_class
    
    def _create_basic_features(self):
        """Create basic derived features"""
        print("Creating basic features...")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Year_Built',
                       'Floor_No', 'Total_Floors', 'Age_of_Property',
                       'Nearby_Schools', 'Nearby_Hospitals', 'Parking_Space']
        
        for col in numeric_cols:
            if col in self.data.columns:
                # Ensure numeric type
                if self.data[col].dtype != np.number:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Ensure Public_Transport_Accessibility is numeric
        if 'Public_Transport_Accessibility' in self.data.columns:
            if self.data['Public_Transport_Accessibility'].dtype == 'object':
                # Extract first number from string
                self.data['Public_Transport_Accessibility'] = (
                    self.data['Public_Transport_Accessibility']
                    .astype(str)
                    .str.extract('(\d+)', expand=False)
                    .astype(float)
                )
            self.data['Public_Transport_Accessibility'] = pd.to_numeric(
                self.data['Public_Transport_Accessibility'], errors='coerce'
            )
            # Fill missing values
            median_val = self.data['Public_Transport_Accessibility'].median()
            self.data['Public_Transport_Accessibility'].fillna(median_val, inplace=True)
        
        # Create Age_of_Property if not exists
        if 'Age_of_Property' not in self.data.columns and 'Year_Built' in self.data.columns:
            current_year = datetime.now().year
            self.data['Age_of_Property'] = current_year - self.data['Year_Built']
            # Cap at reasonable values
            self.data['Age_of_Property'] = self.data['Age_of_Property'].clip(0, 100)
            print("  Created Age_of_Property")
        
        # Create Price_per_SqFt if not exists
        if 'Price_per_SqFt' not in self.data.columns:
            if 'Price_in_Lakhs' in self.data.columns and 'Size_in_SqFt' in self.data.columns:
                self.data['Price_per_SqFt'] = (self.data['Price_in_Lakhs'] * 100000) / self.data['Size_in_SqFt']
                # Handle division by zero
                self.data['Price_per_SqFt'].replace([np.inf, -np.inf], np.nan, inplace=True)
                # Fill missing values
                median_pps = self.data['Price_per_SqFt'].median()
                self.data['Price_per_SqFt'].fillna(median_pps, inplace=True)
                print(f"  Created Price_per_SqFt (median: {median_pps:.2f})")
    
    def _create_simple_investment_features(self):
        """Create simplified investment features without qcut"""
        print("Creating simplified investment features...")
        
        # Create score columns using simple binning
        score_columns = ['Nearby_Schools', 'Nearby_Hospitals', 
                        'Public_Transport_Accessibility', 'Parking_Space']
        
        for col in score_columns:
            if col in self.data.columns:
                score_col = col.replace('_', '') + 'Score'
                
                # Use simple binning instead of qcut
                # Create 5 equal-width bins
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                bin_width = (max_val - min_val) / 5
                
                # Create bins
                bins = [min_val + i * bin_width for i in range(6)]
                bins[-1] = max_val + 0.1  # Ensure max value is included
                
                # Create labels
                labels = [1, 2, 3, 4, 5]
                
                # Apply binning
                self.data[score_col] = pd.cut(
                    self.data[col],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                ).astype(int)
                
                print(f"  Created {score_col}")
        
        # Create composite Infrastructure Score
        score_cols = [col for col in self.data.columns if 'Score' in col]
        if score_cols:
            self.data['Infrastructure_Score'] = self.data[score_cols].mean(axis=1)
            print("  Created Infrastructure_Score")
        
        # Amenities count
        if 'Amenities' in self.data.columns:
            self.data['Amenities'] = self.data['Amenities'].astype(str)
            
            # Count number of amenities (comma-separated)
            def count_amenities(x):
                if pd.isna(x) or x.lower() == 'nan' or x == '':
                    return 0
                # Split by comma and count
                return len(str(x).split(','))
            
            self.data['Amenities_Count'] = self.data['Amenities'].apply(count_amenities)
            print("  Created Amenities_Count")
    
    def _create_target_variables(self):
        """Create target variables"""
        print("Creating target variables...")
        
        # REGRESSION TARGET: Future Price after 5 years
        if 'Price_in_Lakhs' in self.data.columns:
            print("  Creating regression target...")
            
            # Use fixed appreciation rate of 8% per year
            appreciation_rate = 0.08  # 8% annual appreciation
            
            # Calculate future price
            self.data['Future_Price_5Y'] = self.data['Price_in_Lakhs'] * ((1 + appreciation_rate) ** 5)
            
            # Add some variation based on city if available
            if 'City' in self.data.columns:
                city_groups = self.data.groupby('City')
                
                for city, group in city_groups:
                    if len(group) > 10:  # Only for cities with enough data
                        city_mean = group['Price_in_Lakhs'].mean()
                        city_std = group['Price_in_Lakhs'].std()
                        
                        if city_std > 0:
                            # Add city-specific variation
                            city_idx = self.data[self.data['City'] == city].index
                            city_variation = np.random.normal(0, city_std * 0.1, len(city_idx))
                            self.data.loc[city_idx, 'Future_Price_5Y'] *= (1 + city_variation / self.data.loc[city_idx, 'Price_in_Lakhs'])
            
            print(f"    Future price range: ₹{self.data['Future_Price_5Y'].min():.2f} - ₹{self.data['Future_Price_5Y'].max():.2f} L")
        
        # CLASSIFICATION TARGET: Good Investment
        print("  Creating classification target...")
        
        # Initialize investment score
        self.data['Investment_Score'] = 0
        
        # Rule 1: Price below city median (if City available)
        if 'City' in self.data.columns and 'Price_in_Lakhs' in self.data.columns:
            city_medians = self.data.groupby('City')['Price_in_Lakhs'].transform('median')
            self.data['Investment_Score'] += (self.data['Price_in_Lakhs'] <= city_medians).astype(int) * 2
            print("    Added price vs city median rule")
        
        # Rule 2: Price per SqFt below city median
        if 'City' in self.data.columns and 'Price_per_SqFt' in self.data.columns:
            city_pps_medians = self.data.groupby('City')['Price_per_SqFt'].transform('median')
            self.data['Investment_Score'] += (self.data['Price_per_SqFt'] <= city_pps_medians).astype(int) * 2
            print("    Added price per sqft rule")
        
        # Rule 3: Good BHK (2-4 is optimal)
        if 'BHK' in self.data.columns:
            self.data['Investment_Score'] += ((self.data['BHK'] >= 2) & (self.data['BHK'] <= 4)).astype(int)
            print("    Added BHK rule")
        
        # Rule 4: Newer property (<= 10 years)
        if 'Age_of_Property' in self.data.columns:
            self.data['Investment_Score'] += (self.data['Age_of_Property'] <= 10).astype(int)
            print("    Added property age rule")
        
        # Rule 5: Good infrastructure (score >= 3)
        if 'Infrastructure_Score' in self.data.columns:
            self.data['Investment_Score'] += (self.data['Infrastructure_Score'] >= 3).astype(int)
            print("    Added infrastructure rule")
        
        # Rule 6: Good amenities (count >= 3)
        if 'Amenities_Count' in self.data.columns:
            self.data['Investment_Score'] += (self.data['Amenities_Count'] >= 3).astype(int)
            print("    Added amenities rule")
        
        # Create binary classification (threshold = 5 out of max possible score)
        max_possible_score = 10  # Adjust based on rules added
        self.data['Good_Investment'] = (self.data['Investment_Score'] >= (max_possible_score // 2)).astype(int)
        
        # Display distribution
        good_inv_percent = self.data['Good_Investment'].mean() * 100
        print(f"    Good investments: {good_inv_percent:.1f}% of properties")
    
    def _select_final_features(self):
        """Select final features for modeling"""
        print("\nSelecting final features...")
        
        # List of potential features
        potential_features = [
            # Numerical features
            'BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Age_of_Property',
            'Floor_No', 'Total_Floors', 'Nearby_Schools', 'Nearby_Hospitals',
            'Public_Transport_Accessibility', 'Parking_Space',
            
            # Engineered features
            'NearbySchoolsScore', 'NearbyHospitalsScore',
            'PublicTransportAccessibilityScore', 'ParkingSpaceScore',
            'Infrastructure_Score', 'Amenities_Count',
            
            # Categorical features (will be encoded)
            'Property_Type', 'Furnished_Status', 'City', 'Facing'
        ]
        
        # Only keep features that exist in data
        selected_features = []
        for feat in potential_features:
            if feat in self.data.columns:
                selected_features.append(feat)
        
        print(f"  Selected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"    - {feat}")
        
        # Get feature matrix
        X = self.data[selected_features].copy()
        
        # Get target variables
        if 'Future_Price_5Y' not in self.data.columns:
            raise ValueError("Future_Price_5Y target not created")
        if 'Good_Investment' not in self.data.columns:
            raise ValueError("Good_Investment target not created")
        
        y_reg = self.data['Future_Price_5Y'].copy()
        y_class = self.data['Good_Investment'].copy()
        
        return X, y_reg, y_class
    
    def save_features(self, output_path):
        """Save engineered features to file"""
        features_df = pd.concat([
            self.data[[col for col in self.data.columns if col not in ['Future_Price_5Y', 'Good_Investment', 'Investment_Score']]],
            self.data[['Future_Price_5Y', 'Good_Investment']]
        ], axis=1)
        
        features_df.to_csv(output_path, index=False)
        print(f"\n✅ Features saved to: {output_path}")
        return features_df