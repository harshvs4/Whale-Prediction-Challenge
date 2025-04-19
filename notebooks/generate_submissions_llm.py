import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare data for training and prediction"""
    print("Loading data with LLM features...")
    
    # Load training data with LLM features
    train_data = pd.read_csv('../data/llm_train_features_v2.csv')
    test_data = pd.read_csv('../data/llm_test_features_v2.csv')
    
    # Convert categorical variables
    categorical_cols = ['income_level', 'mobile_brand', 'mobile_model', 'gender', 'education']
    for col in categorical_cols:
        if col in train_data.columns:
            # Fill missing values with a placeholder
            train_data[col] = train_data[col].fillna('unknown')
            test_data[col] = test_data[col].fillna('unknown')
            
            # Convert to category
            train_data[col] = train_data[col].astype('category')
            test_data[col] = test_data[col].astype('category')
            
            # Ensure test data categories are a subset of train data categories
            test_data[col] = test_data[col].cat.set_categories(train_data[col].cat.categories)
    
    # Separate features and target
    y = train_data['tgt']
    feature_cols = [col for col in train_data.columns if col not in ['tgt', 'user_id']]
    X = train_data[feature_cols]
    
    # Prepare test data
    test_ids = test_data['user_id']
    X_test = test_data[feature_cols]
    
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("\nFeature types:")
    for col in X.columns:
        print(f"{col}: {X[col].dtype}")
    
    return X, y, X_test, test_ids

def train_xgboost(X, y):
    """Train XGBoost model with LLM features"""
    print("\nTraining XGBoost model...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define model parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'enable_categorical': True  # Enable categorical feature support
    }
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_pred)
    print(f"XGBoost Validation AUC: {auc_score:.4f}")
    
    return model

def train_lightgbm(X, y):
    """Train LightGBM model with LLM features"""
    print("\nTraining LightGBM model...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define model parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42
    }
    
    # Train model
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_pred)
    print(f"LightGBM Validation AUC: {auc_score:.4f}")
    
    return model

def save_submission(predictions, test_ids, model_name):
    """Save predictions to a submission file"""
    submission = pd.DataFrame({
        'user_id': test_ids,
        'tgt': predictions  # Changed from 'whale' to 'tgt'
    })
    
    filename = f'../submissions/submission_{model_name}_llm.csv'
    submission.to_csv(filename, index=False)
    print(f"Saved submission to {filename}")

def main():
    """Main execution function"""
    # Create submissions directory if it doesn't exist
    os.makedirs('../submissions', exist_ok=True)
    
    # Load data
    X, y, X_test, test_ids = load_data()
    
    # Train models and generate predictions
    xgb_model = train_xgboost(X, y)
    lgb_model = train_lightgbm(X, y)
    
    # Generate predictions
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    
    # Save individual model predictions
    save_submission(xgb_pred, test_ids, 'xgboost')
    save_submission(lgb_pred, test_ids, 'lightgbm')
    
    # Create and save ensemble predictions
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred
    save_submission(ensemble_pred, test_ids, 'ensemble')
    
    print("\nAll submissions generated successfully!")

if __name__ == "__main__":
    main() 