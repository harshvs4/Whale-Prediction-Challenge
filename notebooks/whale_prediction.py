import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Data Loading
print("Loading data...")
profile_data = pd.read_csv('../data/profile.csv')
transaction_data = pd.read_csv('../data/trx_data.csv')
train_labels = pd.read_csv('../data/train_label.csv')

# Display basic information
print("Data shapes:")
print(f"Profile data: {profile_data.shape}")
print(f"Transaction data: {transaction_data.shape}")
print(f"Train labels: {train_labels.shape}")

# EDA and Data Preprocessing
print("\nProfile data info:")
print(profile_data.info())
print("\nProfile data sample:")
print(profile_data.head())

print("\nTransaction data info:")
print(transaction_data.info())
print("\nTransaction data sample:")
print(transaction_data.head())

print("\nTrain labels info:")
print(train_labels.info())
print("\nClass distribution:")
print(train_labels['tgt'].value_counts(normalize=True))

# Preprocess the transaction data
print("\nPreprocessing transaction data...")
# Convert transaction_time to datetime
transaction_data['transaction_time'] = pd.to_datetime(transaction_data['transaction_time'])
# Filter data to April-May 2022 timeframe
transaction_data = transaction_data[(transaction_data['transaction_time'] >= '2022-04-01') & 
                                    (transaction_data['transaction_time'] <= '2022-05-31')]

# Feature Engineering
print("\nFeature engineering...")
# Group by user_id to create aggregate features
trx_agg = transaction_data.groupby('user_id').agg(
    total_transactions=('transaction_time', 'count'),
    total_gtv=('gtv', 'sum'),
    avg_gtv=('gtv', 'mean'),
    min_gtv=('gtv', 'min'),
    max_gtv=('gtv', 'max'),
    std_gtv=('gtv', 'std'),
    first_transaction=('transaction_time', 'min'),
    last_transaction=('transaction_time', 'max')
).reset_index()

# Calculate days between first and last transaction
trx_agg['transaction_span_days'] = (trx_agg['last_transaction'] - trx_agg['first_transaction']).dt.days
# For those with only one transaction, set span to 0
trx_agg['transaction_span_days'] = trx_agg['transaction_span_days'].fillna(0)

# Transaction type features (BUY/SELL)
buy_agg = transaction_data[transaction_data['transaction_type'] == 'BUY'].groupby('user_id').agg(
    buy_transactions=('transaction_time', 'count'),
    buy_total_gtv=('gtv', 'sum'),
    buy_avg_gtv=('gtv', 'mean')
).reset_index()

sell_agg = transaction_data[transaction_data['transaction_type'] == 'SELL'].groupby('user_id').agg(
    sell_transactions=('transaction_time', 'count'),
    sell_total_gtv=('gtv', 'sum'),
    sell_avg_gtv=('gtv', 'mean')
).reset_index()

# Asset type features (crypto/others)
asset_type_counts = pd.get_dummies(transaction_data[['user_id', 'asset_type']], columns=['asset_type'])
asset_type_agg = asset_type_counts.groupby('user_id').sum().reset_index()

# Time-based features - extract day, hour, weekday
transaction_data['day'] = transaction_data['transaction_time'].dt.day
transaction_data['hour'] = transaction_data['transaction_time'].dt.hour
transaction_data['weekday'] = transaction_data['transaction_time'].dt.weekday

# Weekly patterns
weekday_agg = transaction_data.groupby(['user_id', 'weekday']).size().unstack(fill_value=0)
weekday_agg.columns = [f'weekday_{i}' for i in weekday_agg.columns]
weekday_agg.reset_index(inplace=True)

# Hourly patterns
hour_agg = transaction_data.groupby(['user_id', 'hour']).size().unstack(fill_value=0)
hour_agg.columns = [f'hour_{i}' for i in hour_agg.columns]
hour_agg.reset_index(inplace=True)

# Monthly patterns - early vs late month
transaction_data['early_month'] = transaction_data['day'] <= 15
early_month_agg = transaction_data.groupby(['user_id', 'early_month']).size().unstack(fill_value=0)
if True in early_month_agg.columns:
    early_month_agg.rename(columns={True: 'early_month_trx', False: 'late_month_trx'}, inplace=True)
else:
    early_month_agg['early_month_trx'] = 0
    early_month_agg['late_month_trx'] = 0
early_month_agg.reset_index(inplace=True)

# Merge all transaction features
print("Merging transaction features...")
features_list = [trx_agg, buy_agg, sell_agg, asset_type_agg, weekday_agg, hour_agg, early_month_agg]
transaction_features = features_list[0]
for df in features_list[1:]:
    transaction_features = pd.merge(transaction_features, df, on='user_id', how='left')

# Fill NAs in transaction features
transaction_features = transaction_features.fillna(0)

# Calculate additional derived features
transaction_features['buy_sell_ratio'] = transaction_features['buy_transactions'] / (transaction_features['sell_transactions'] + 1)
transaction_features['transaction_frequency'] = transaction_features['total_transactions'] / (transaction_features['transaction_span_days'] + 1)
transaction_features['weekend_ratio'] = (transaction_features['weekday_5'] + transaction_features['weekday_6']) / \
                                        (transaction_features['total_transactions'] + 1)
transaction_features['night_ratio'] = (transaction_features['hour_23'] + transaction_features['hour_0'] + 
                                      transaction_features['hour_1'] + transaction_features['hour_2']) / \
                                     (transaction_features['total_transactions'] + 1)

# Process user profile data
print("Processing user profile data...")
# Handle missing values in the profile data
profile_data['age_in_year'] = profile_data['age_in_year'].fillna(profile_data['age_in_year'].median())

# Extract brand information as potential proxy for wealth
profile_data['is_premium_brand'] = profile_data['mobile_brand_name'].isin(['Apple', 'Samsung', 'OnePlus', 'Google'])

# Merge profile data with transaction features
print("Merging profile and transaction data...")
combined_features = pd.merge(profile_data, transaction_features, on='user_id', how='inner')

# Create final training dataset by merging with labels
print("Creating final training dataset...")
train_data = pd.merge(combined_features, train_labels, on='user_id', how='inner')

# Optional: print summary of final training data
print(f"\nFinal training data shape: {train_data.shape}")
print(train_data.columns.tolist())

# Split for model validation
X = train_data.drop(['user_id', 'tgt'], axis=1)
y = train_data['tgt']

# One-hot encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# LightGBM Model
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

# Optimize LightGBM hyperparameters
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'num_leaves': [31, 50, 100],
    'max_depth': [5, 7, 9]
}

# Use a subset of data for faster hyperparameter tuning
X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42, stratify=y_train)

# Preprocess the sampled data
X_sample_processed = preprocessor.fit_transform(X_sample)

# Grid search
grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_sample_processed, y_sample)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best LightGBM parameters: {best_params}")

# Train final model with best parameters on full training data
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

best_lgb_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')
best_lgb_model.fit(X_train_processed, y_train)

# Evaluate model
y_pred_proba = best_lgb_model.predict_proba(X_val_processed)[:, 1]
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"\nValidation AUC: {auc_score:.4f}")

# Getting feature importance
feature_names = (
    numerical_cols + 
    preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 important features:")
print(feature_importance.head(20))

# Generate predictions for all users
print("\nGenerating predictions for all users...")
# Get all user IDs from profile data
all_users = profile_data['user_id'].unique()
print(f"Total users for prediction: {len(all_users)}")

# Create features for all users
all_features = combined_features[combined_features['user_id'].isin(all_users)]

# Prepare data for prediction
X_all = all_features.drop(['user_id'], axis=1)
X_all_processed = preprocessor.transform(X_all)

# Generate predictions
all_pred_proba = best_lgb_model.predict_proba(X_all_processed)[:, 1]
submission = pd.DataFrame({
    'user_id': all_features['user_id'],
    'pred_prob': all_pred_proba
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!") 