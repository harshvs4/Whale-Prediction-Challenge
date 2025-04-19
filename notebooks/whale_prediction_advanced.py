import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import os
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the enhanced datasets"""
    print("Loading enhanced datasets...")
    train_data = pd.read_csv('../data/train_enhanced.csv')
    test_data = pd.read_csv('../data/test_enhanced.csv')
    
    # Define features to use
    base_features = [
        'min_gtv', 'max_gtv', 'mean_gtv', 'std_gtv', 'total_gtv',
        'buy_total_gtv', 'sell_total_gtv', 'buy_sell_ratio',
        'transaction_frequency', 'transaction_span_days'
    ]
    
    llm_features = [
        'openai_whale_score', 'gemini_whale_score', 'combined_llm_score',
        'llm_score_std', 'llm_score_mean', 'llm_score_diff'
    ]
    
    all_features = base_features + llm_features
    
    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(
        train_data[all_features], train_data['is_whale'],
        test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val, test_data[all_features], test_data['user_id']

def train_models(X_train, y_train, X_val, y_val):
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
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
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
    xgb_val = xgb.DMatrix(X_val, y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=1000,
        evals=[(xgb_train, 'train'), (xgb_val, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return lgb_model, xgb_model

def evaluate_models(lgb_model, xgb_model, X_val, y_val):
    """Evaluate models on validation set"""
    print("Evaluating models...")
    
    # LightGBM predictions
    lgb_preds = lgb_model.predict(X_val)
    lgb_auc = roc_auc_score(y_val, lgb_preds)
    
    # XGBoost predictions
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_val))
    xgb_auc = roc_auc_score(y_val, xgb_preds)
    
    # Ensemble predictions
    ensemble_preds = 0.6 * lgb_preds + 0.4 * xgb_preds
    ensemble_auc = roc_auc_score(y_val, ensemble_preds)
    
    print(f"LightGBM AUC: {lgb_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    print(f"Ensemble AUC: {ensemble_auc:.4f}")
    
    return lgb_auc, xgb_auc, ensemble_auc

def generate_submission(lgb_model, xgb_model, X_test, test_ids):
    """Generate submission file"""
    print("Generating submission...")
    
    # Get predictions from both models
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X_test))
    
    # Ensemble predictions
    ensemble_preds = 0.6 * lgb_preds + 0.4 * xgb_preds
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'user_id': test_ids,
        'pred_prob': ensemble_preds
    })
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'../submissions/submission_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission saved to {submission_path}")
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Min prediction: {ensemble_preds.min():.4f}")
    print(f"Max prediction: {ensemble_preds.max():.4f}")
    print(f"Mean prediction: {ensemble_preds.mean():.4f}")
    print(f"Median prediction: {np.median(ensemble_preds):.4f}")
    print(f"Predicted whales (>0.5): {sum(ensemble_preds > 0.5)} users ({sum(ensemble_preds > 0.5)/len(ensemble_preds)*100:.2f}%)")

def main():
    # Create necessary directories
    os.makedirs('../submissions', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, X_test, test_ids = load_and_prepare_data()
    
    # Train models
    lgb_model, xgb_model = train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    lgb_auc, xgb_auc, ensemble_auc = evaluate_models(lgb_model, xgb_model, X_val, y_val)
    
    # Generate submission
    generate_submission(lgb_model, xgb_model, X_test, test_ids)
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'../models/lgb_model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    with open(f'../models/xgb_model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

if __name__ == "__main__":
    main() 