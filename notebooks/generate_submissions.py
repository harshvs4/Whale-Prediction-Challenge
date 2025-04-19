import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data():
    """Load and prepare data"""
    print("Loading data...")
    train_data = pd.read_csv('../data/preprocessed_train.csv')
    test_data = pd.read_csv('../data/preprocessed_test.csv')
    advanced_train = pd.read_csv('../data/advanced_train_features.csv')
    advanced_test = pd.read_csv('../data/advanced_test_features.csv')
    
    # Combine original and advanced features
    X_train = pd.concat([train_data, advanced_train], axis=1)
    X_test = pd.concat([test_data, advanced_test], axis=1)
    
    # Remove non-feature columns
    feature_cols = [col for col in X_train.columns if col not in ['user_id', 'tgt']]
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
    
    # Convert categorical columns to numeric
    categorical_cols = ['income_level', 'mobile_brand', 'mobile_model', 'gender', 'education']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X_train.columns:
            # Get unique values from both train and test
            unique_values = pd.concat([X_train[col], X_test[col]]).unique()
            le = LabelEncoder()
            le.fit(unique_values.astype(str))
            
            # Transform train and test data
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # Convert all columns to float
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    y_train = train_data['tgt']
    
    return X_train, y_train, X_test, test_data['user_id']

def train_models(X_train, y_train):
    """Train LightGBM and XGBoost models"""
    print("Training models...")
    
    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, 
                         valid_sets=[lgb_train], callbacks=[lgb.early_stopping(50)])
    
    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'tree_method': 'hist'
    }
    
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=1000,
                         evals=[(xgb_train, 'train')], early_stopping_rounds=50)
    
    return lgb_model, xgb_model

def generate_predictions(lgb_model, xgb_model, X_test, user_ids):
    """Generate predictions for all models"""
    print("Generating predictions...")
    
    # Get predictions from both models
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_test))
    
    # Ensemble predictions (weighted average)
    ensemble_preds = 0.6 * lgb_preds + 0.4 * xgb_preds
    
    # Create submission DataFrames
    submissions = {
        'submission_lightgbm.csv': lgb_preds,
        'submission_xgboost.csv': xgb_preds,
        'submission_ensemble.csv': ensemble_preds
    }
    
    return submissions

def save_submissions(submissions, user_ids):
    """Save all submission files"""
    print("Saving submissions...")
    
    # Create submissions directory if it doesn't exist
    os.makedirs('../submissions', exist_ok=True)
    
    for filename, predictions in submissions.items():
        submission = pd.DataFrame({
            'user_id': user_ids,
            'pred_prob': predictions  # Ensure the column name is 'pred_prob'
        })
        submission.to_csv(f'../submissions/{filename}', index=False)
        print(f"Saved {filename}")

def main():
    # Load data
    X_train, y_train, X_test, user_ids = load_data()
    
    # Train models
    lgb_model, xgb_model = train_models(X_train, y_train)
    
    # Generate predictions
    submissions = generate_predictions(lgb_model, xgb_model, X_test, user_ids)
    
    # Save submissions
    save_submissions(submissions, user_ids)
    
    # Calculate and print validation scores
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    lgb_model_val, xgb_model_val = train_models(X_train_val, y_train_val)
    
    lgb_val_preds = lgb_model_val.predict(X_val)
    xgb_val_preds = xgb_model_val.predict(xgb.DMatrix(X_val))
    ensemble_val_preds = 0.6 * lgb_val_preds + 0.4 * xgb_val_preds
    
    print("\nValidation Scores:")
    print(f"LightGBM AUC Score: {roc_auc_score(y_val, lgb_val_preds):.4f}")
    print(f"XGBoost AUC Score: {roc_auc_score(y_val, xgb_val_preds):.4f}")
    print(f"Ensemble AUC Score: {roc_auc_score(y_val, ensemble_val_preds):.4f}")
    
    print("\nSubmission files generated successfully!")

if __name__ == "__main__":
    main() 