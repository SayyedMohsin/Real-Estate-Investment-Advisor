import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, r2_score, confusion_matrix)
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, X, y_class, y_reg):
        self.X = X.copy()
        self.y_class = y_class.copy()
        self.y_reg = y_reg.copy()
        
        print(f"\nInitializing Model Trainer...")
        print(f"Features shape: {self.X.shape}")
        print(f"Classification target shape: {self.y_class.shape}")
        print(f"Regression target shape: {self.y_reg.shape}")
        
        # Identify categorical and numerical columns
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\nCategorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        
        # Split data
        self.split_data()
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        print(f"Models directory ready")
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("file:///C:/JN/Real-Estate-Investment-Advisor/mlruns")
            
            # Set experiment name
            mlflow.set_experiment("real_estate_investment")
            
            print("✅ MLflow setup completed")
            print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"   Experiment: {mlflow.get_experiment_by_name('real_estate_investment').name}")
            
        except Exception as e:
            print(f"⚠️ MLflow setup warning: {str(e)}")
            print("   Continuing without MLflow...")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        
        # Split for classification
        self.X_train, self.X_test, self.y_train_class, self.y_test_class = train_test_split(
            self.X, self.y_class, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.y_class
        )
        
        # Split for regression (using same indices)
        _, _, self.y_train_reg, self.y_test_reg = train_test_split(
            self.X, self.y_reg, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        
        # Show class distribution
        print(f"\n  Class distribution in training set:")
        class_dist = pd.Series(self.y_train_class).value_counts()
        for class_val, count in class_dist.items():
            percentage = (count / len(self.y_train_class)) * 100
            label = "Good Investment" if class_val == 1 else "Not Good Investment"
            print(f"    {label}: {count} ({percentage:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train_class, self.y_test_class, self.y_train_reg, self.y_test_reg
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        print("\nCreating preprocessing pipeline...")
        
        # Numerical transformer
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        
        print(f"  Preprocessor created with {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical features")
        return preprocessor
    
    def train_classification_model(self, model_type='random_forest'):
        """Train classification model"""
        
        print(f"\n{'='*50}")
        print(f"TRAINING CLASSIFICATION MODEL: {model_type.upper()}")
        print(f"{'='*50}")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"classification_{model_type}"):
                
                # Create preprocessing pipeline
                preprocessor = self.create_preprocessor()
                
                # Define model
                if model_type == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                elif model_type == 'logistic':
                    model = LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        class_weight='balanced',
                        C=1.0
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Create full pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                print(f"\nTraining {model_type} classifier...")
                
                # Train model
                pipeline.fit(self.X_train, self.y_train_class)
                
                # Make predictions
                y_pred = pipeline.predict(self.X_test)
                y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test_class, y_pred)
                precision = precision_score(self.y_test_class, y_pred)
                recall = recall_score(self.y_test_class, y_pred)
                f1 = f1_score(self.y_test_class, y_pred)
                roc_auc = roc_auc_score(self.y_test_class, y_pred_proba)
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test_class, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'true_positive': tp,
                    'true_negative': tn,
                    'false_positive': fp,
                    'false_negative': fn
                }
                
                # Log parameters to MLflow
                mlflow.log_params({
                    'model_type': model_type,
                    'num_features': len(self.X.columns),
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test)
                })
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(pipeline, f"{model_type}_classifier")
                
                # Display results
                print(f"\n📊 CLASSIFICATION RESULTS:")
                print(f"   Accuracy:  {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall:    {recall:.4f}")
                print(f"   F1 Score:  {f1:.4f}")
                print(f"   ROC AUC:   {roc_auc:.4f}")
                print(f"\n   Confusion Matrix:")
                print(f"   TP: {tp}, FP: {fp}")
                print(f"   FN: {fn}, TN: {tn}")
                
                # Cross-validation scores
                print(f"\n   Cross-validation (5-fold):")
                cv_scores = cross_val_score(pipeline, self.X, self.y_class, cv=5, scoring='f1')
                print(f"   F1 Scores: {cv_scores}")
                print(f"   Mean F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                return pipeline, metrics
                
        except Exception as e:
            print(f"❌ Error training classification model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_regression_model(self, model_type='random_forest'):
        """Train regression model"""
        
        print(f"\n{'='*50}")
        print(f"TRAINING REGRESSION MODEL: {model_type.upper()}")
        print(f"{'='*50}")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"regression_{model_type}"):
                
                # Create preprocessing pipeline
                preprocessor = self.create_preprocessor()
                
                # Define model
                if model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == 'linear':
                    model = LinearRegression()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Create full pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                print(f"\nTraining {model_type} regressor...")
                
                # Train model
                pipeline.fit(self.X_train, self.y_train_reg)
                
                # Make predictions
                y_pred = pipeline.predict(self.X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                mae = mean_absolute_error(self.y_test_reg, y_pred)
                r2 = r2_score(self.y_test_reg, y_pred)
                
                # Calculate MAPE (handle zero values)
                y_test_nonzero = self.y_test_reg[self.y_test_reg != 0]
                y_pred_nonzero = y_pred[self.y_test_reg != 0]
                
                if len(y_test_nonzero) > 0:
                    mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
                else:
                    mape = 0
                
                metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape
                }
                
                # Log parameters to MLflow
                mlflow.log_params({
                    'model_type': model_type,
                    'num_features': len(self.X.columns),
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test)
                })
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(pipeline, f"{model_type}_regressor")
                
                # Display results
                print(f"\n📊 REGRESSION RESULTS:")
                print(f"   RMSE:  ₹{rmse:.2f} L")
                print(f"   MAE:   ₹{mae:.2f} L")
                print(f"   R²:    {r2:.4f}")
                print(f"   MAPE:  {mape:.2f}%")
                
                # Display sample predictions
                print(f"\n   Sample predictions (first 5):")
                for i in range(min(5, len(y_pred))):
                    print(f"   Actual: ₹{self.y_test_reg.iloc[i]:.2f} L, "
                          f"Predicted: ₹{y_pred[i]:.2f} L, "
                          f"Error: ₹{abs(self.y_test_reg.iloc[i] - y_pred[i]):.2f} L")
                
                # Cross-validation scores
                print(f"\n   Cross-validation (5-fold) R² scores:")
                cv_scores = cross_val_score(pipeline, self.X, self.y_reg, cv=5, scoring='r2')
                print(f"   R² Scores: {cv_scores}")
                print(f"   Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                return pipeline, metrics
                
        except Exception as e:
            print(f"❌ Error training regression model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_models(self, class_model, reg_model, class_type='random_forest', reg_type='random_forest'):
        """Save trained models"""
        print(f"\n{'='*50}")
        print("SAVING MODELS")
        print(f"{'='*50}")
        
        try:
            # Save classification model
            class_path = f'models/classification_{class_type}.pkl'
            joblib.dump(class_model, class_path)
            print(f"✅ Classification model saved: {class_path}")
            
            # Save regression model
            reg_path = f'models/regression_{reg_type}.pkl'
            joblib.dump(reg_model, reg_path)
            print(f"✅ Regression model saved: {reg_path}")
            
            # Save feature columns
            feature_columns_path = 'models/feature_columns.pkl'
            joblib.dump(list(self.X.columns), feature_columns_path)
            print(f"✅ Feature columns saved: {feature_columns_path}")
            
            # Save preprocessing info
            preprocessing_info = {
                'categorical_cols': self.categorical_cols,
                'numerical_cols': self.numerical_cols,
                'all_features': list(self.X.columns)
            }
            preprocessing_path = 'models/preprocessing_info.pkl'
            joblib.dump(preprocessing_info, preprocessing_path)
            print(f"✅ Preprocessing info saved: {preprocessing_path}")
            
            return class_path, reg_path
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return None, None
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print(f"\n{'='*60}")
        print("STARTING COMPLETE TRAINING PIPELINE")
        print(f"{'='*60}")
        
        # Train classification model
        print(f"\n🚀 Training Classification Model...")
        class_model, class_metrics = self.train_classification_model('random_forest')
        
        if class_model is None:
            print("❌ Failed to train classification model")
            return False
        
        # Train regression model
        print(f"\n🚀 Training Regression Model...")
        reg_model, reg_metrics = self.train_regression_model('random_forest')
        
        if reg_model is None:
            print("❌ Failed to train regression model")
            return False
        
        # Save models
        print(f"\n💾 Saving Models...")
        self.save_models(class_model, reg_model)
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print(f"{'='*60}")
        
        print(f"\n📊 FINAL MODEL PERFORMANCE:")
        print(f"{'-'*40}")
        
        if class_metrics:
            print(f"CLASSIFICATION (Good Investment):")
            print(f"  Accuracy:  {class_metrics['accuracy']:.4f}")
            print(f"  F1 Score:  {class_metrics['f1_score']:.4f}")
            print(f"  ROC AUC:   {class_metrics['roc_auc']:.4f}")
        
        if reg_metrics:
            print(f"\nREGRESSION (Future Price):")
            print(f"  RMSE:      ₹{reg_metrics['rmse']:.2f} L")
            print(f"  R² Score:  {reg_metrics['r2']:.4f}")
            print(f"  MAPE:      {reg_metrics['mape']:.2f}%")
        
        print(f"\n📁 Files saved in 'models/' directory:")
        print(f"  - classification_random_forest.pkl")
        print(f"  - regression_random_forest.pkl")
        print(f"  - feature_columns.pkl")
        print(f"  - preprocessing_info.pkl")
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Run Streamlit app: streamlit run app_simple.py")
        print(f"  2. View MLflow UI: mlflow ui")
        print(f"  3. Check models in 'models/' directory")
        
        return True