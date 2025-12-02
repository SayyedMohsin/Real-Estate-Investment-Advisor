import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        print(f"Dataset loaded: {self.data.shape}")
        
    def load_data(self):
        """Load and display basic data info"""
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData Types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
        # Show basic statistics
        if 'Price_in_Lakhs' in self.data.columns:
            print(f"\nPrice Statistics:")
            print(f"Min: ₹{self.data['Price_in_Lakhs'].min():.2f} L")
            print(f"Max: ₹{self.data['Price_in_Lakhs'].max():.2f} L")
            print(f"Mean: ₹{self.data['Price_in_Lakhs'].mean():.2f} L")
            print(f"Median: ₹{self.data['Price_in_Lakhs'].median():.2f} L")
        
        return self.data
    
    def clean_data(self):
        """Main cleaning function"""
        print("\nStarting data cleaning...")
        
        # Step 1: Convert problematic columns to numeric
        self._convert_columns_to_numeric()
        
        # Step 2: Clean categorical columns
        self._clean_categorical_columns()
        
        # Step 3: Handle missing values
        self._handle_missing_values()
        
        # Step 4: Remove duplicates
        self._remove_duplicates()
        
        # Step 5: Handle unrealistic values
        self._handle_unrealistic_values()
        
        # Step 6: Create additional useful columns
        self._create_additional_columns()
        
        print(f"\n✅ Final dataset shape: {self.data.shape}")
        return self.data
    
    def _convert_columns_to_numeric(self):
        """Convert columns that should be numeric"""
        print("Converting columns to numeric...")
        
        # Columns that should be numeric
        numeric_columns = [
            'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
            'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
            'Nearby_Schools', 'Nearby_Hospitals', 'Parking_Space'
        ]
        
        for col in numeric_columns:
            if col in self.data.columns:
                # Try to convert to numeric, coerce errors to NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                print(f"  Converted {col} to numeric")
        
        # Special handling for Public_Transport_Accessibility
        if 'Public_Transport_Accessibility' in self.data.columns:
            # Check if it's already numeric
            if self.data['Public_Transport_Accessibility'].dtype == 'object':
                # Try to extract numeric values
                self.data['Public_Transport_Accessibility'] = (
                    self.data['Public_Transport_Accessibility']
                    .astype(str)
                    .str.extract('(\d+)')
                    .astype(float)
                )
            self.data['Public_Transport_Accessibility'] = pd.to_numeric(
                self.data['Public_Transport_Accessibility'], errors='coerce'
            )
            print(f"  Converted Public_Transport_Accessibility to numeric")
    
    def _clean_categorical_columns(self):
        """Clean categorical columns"""
        print("\nCleaning categorical columns...")
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Convert to string and strip whitespace
            self.data[col] = self.data[col].astype(str).str.strip()
            
            # Replace 'nan' strings with actual NaN
            self.data[col] = self.data[col].replace(['nan', 'NaN', 'None', 'none', ''], np.nan)
            
            # Count unique values
            unique_count = self.data[col].nunique()
            print(f"  {col}: {unique_count} unique values")
    
    def _handle_missing_values(self):
        """Handle missing values"""
        print("\nHandling missing values...")
        
        # Fill numerical columns with median
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                print(f"  Filled {missing_count} missing values in {col} with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                mode_val = self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown'
                self.data[col].fillna(mode_val, inplace=True)
                print(f"  Filled {missing_count} missing values in {col} with mode: {mode_val}")
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        initial_rows = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.data)
        print(f"\nRemoved {removed_rows} duplicate rows")
    
    def _handle_unrealistic_values(self):
        """Handle unrealistic values in data"""
        print("\nHandling unrealistic values...")
        
        # Handle Price_in_Lakhs
        if 'Price_in_Lakhs' in self.data.columns:
            # Remove negative prices and extremely high prices
            initial_count = len(self.data)
            self.data = self.data[(self.data['Price_in_Lakhs'] > 0) & 
                                 (self.data['Price_in_Lakhs'] <= 10000)]  # Max 10000 Lakhs
            removed = initial_count - len(self.data)
            if removed > 0:
                print(f"  Removed {removed} records with unrealistic prices")
        
        # Handle Size_in_SqFt
        if 'Size_in_SqFt' in self.data.columns:
            initial_count = len(self.data)
            self.data = self.data[(self.data['Size_in_SqFt'] > 100) & 
                                 (self.data['Size_in_SqFt'] < 10000)]  # Reasonable size range
            removed = initial_count - len(self.data)
            if removed > 0:
                print(f"  Removed {removed} records with unrealistic sizes")
        
        # Handle BHK
        if 'BHK' in self.data.columns:
            initial_count = len(self.data)
            self.data = self.data[(self.data['BHK'] >= 1) & (self.data['BHK'] <= 10)]
            removed = initial_count - len(self.data)
            if removed > 0:
                print(f"  Removed {removed} records with unrealistic BHK")
    
    def _create_additional_columns(self):
        """Create additional useful columns"""
        print("\nCreating additional columns...")
        
        # Ensure Price_per_SqFt exists
        if 'Price_per_SqFt' not in self.data.columns:
            if 'Price_in_Lakhs' in self.data.columns and 'Size_in_SqFt' in self.data.columns:
                self.data['Price_per_SqFt'] = (self.data['Price_in_Lakhs'] * 100000) / self.data['Size_in_SqFt']
                print("  Created Price_per_SqFt column")
        
        # Cap Price_per_SqFt at reasonable values
        if 'Price_per_SqFt' in self.data.columns:
            pps_99th = self.data['Price_per_SqFt'].quantile(0.99)
            self.data['Price_per_SqFt'] = self.data['Price_per_SqFt'].clip(upper=pps_99th)
            print(f"  Capped Price_per_SqFt at {pps_99th:.2f}")
        
        # Create Price Category
        if 'Price_in_Lakhs' in self.data.columns:
            bins = [0, 50, 100, 200, 500, 1000, float('inf')]
            labels = ['0-50L', '50-100L', '100-200L', '200-500L', '500-1000L', '1000L+']
            self.data['Price_Category'] = pd.cut(self.data['Price_in_Lakhs'], bins=bins, labels=labels)
            print("  Created Price_Category column")
        
        # Create Size Category
        if 'Size_in_SqFt' in self.data.columns:
            bins = [0, 500, 1000, 1500, 2000, 3000, float('inf')]
            labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']
            self.data['Size_Category'] = pd.cut(self.data['Size_in_SqFt'], bins=bins, labels=labels)
            print("  Created Size_Category column")
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to file"""
        self.data.to_csv(output_path, index=False)
        print(f"\n✅ Cleaned data saved to: {output_path}")
        return self.data