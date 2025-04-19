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
import sys
import warnings
import pickle
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings('ignore')

# Add notebooks directory to path for importing the LLM module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_feature_engineering import LLMFeatureEngineer

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create directories for output
create_directory('models')
create_directory('submissions')
create_directory('plots')

# Load Data
print("Loading data...")
profile_data = pd.read_csv('../data/profile.csv')
transaction_data = pd.read_csv('../data/trx_data.csv')
train_labels = pd.read_csv('../data/train_label.csv')
submission_sample = pd.read_csv('../data/submission_sample.csv')

# Display basic information
print("Data shapes:")
print(f"Profile data: {profile_data.shape}")
print(f"Transaction data: {transaction_data.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Submission sample: {submission_sample.shape}")

# Convert transaction_time to datetime
transaction_data['transaction_time'] = pd.to_datetime(transaction_data['transaction_time'])

# Filter data to April-May 2022 timeframe
transaction_data = transaction_data[(transaction_data['transaction_time'] >= '2022-04-01') & 
                                    (transaction_data['transaction_time'] <= '2022-05-31')]

# Traditional Feature Engineering
print("\nTraditional feature engineering...")

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

# Time-based features - extract day, hour, weekday
transaction_data['day'] = transaction_data['transaction_time'].dt.day
transaction_data['hour'] = transaction_data['transaction_time'].dt.hour
transaction_data['weekday'] = transaction_data['transaction_time'].dt.weekday

# Weekly patterns
weekday_agg = transaction_data.groupby(['user_id', 'weekday']).size().unstack(fill_value=0)
weekday_agg.columns = [f'weekday_{i}' for i in weekday_agg.columns]
weekday_agg.reset_index(inplace=True)

# Merge all transaction features
print("Merging transaction features...")
features_list = [trx_agg, buy_agg, sell_agg, weekday_agg]
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

# Process user profile data
print("Processing user profile data...")
# Handle missing values in the profile data
profile_data['age_in_year'] = profile_data['age_in_year'].fillna(profile_data['age_in_year'].median())

# LLM-Enhanced Feature Engineering
print("\nLLM-enhanced feature engineering...")
llm_engineer = LLMFeatureEngineer(profile_data.copy(), transaction_data.copy())
llm_features = llm_engineer.generate_all_llm_features()

# Merge traditional and LLM features
print("Merging traditional and LLM features...")
# First merge profile with traditional transaction features
combined_features = pd.merge(profile_data, transaction_features, on='user_id', how='inner')
# Then add LLM features
llm_subset = llm_features.drop(columns=[col for col in llm_features.columns 
                                        if col in combined_features.columns and col != 'user_id'])
all_features = pd.merge(combined_features, llm_subset, on='user_id', how='inner')

# Create final training dataset by merging with labels
print("Creating final training dataset...")
train_data = pd.merge(all_features, train_labels, on='user_id', how='inner')

# Print summary of final training data
print(f"\nFinal training data shape: {train_data.shape}")

# Split into features and target
X = train_data.drop(['user_id', 'tgt', 'first_transaction', 'last_transaction'], axis=1)
y = train_data['tgt']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Split for model validation
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
        ]), categorical_cols if categorical_cols else [])
    ])

# LightGBM Model
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=50,
    max_depth=7,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# XGBoost Model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='auc'
)

# Preprocessing
print("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Train models
print("Training individual models...")
lgb_model.fit(X_train_processed, y_train)
xgb_model.fit(X_train_processed, y_train)

# Evaluate models
print("Evaluating models...")
lgb_val_preds = lgb_model.predict_proba(X_val_processed)[:, 1]
xgb_val_preds = xgb_model.predict_proba(X_val_processed)[:, 1]

lgb_auc = roc_auc_score(y_val, lgb_val_preds)
xgb_auc = roc_auc_score(y_val, xgb_val_preds)

print(f"LightGBM Validation AUC: {lgb_auc:.4f}")
print(f"XGBoost Validation AUC: {xgb_auc:.4f}")

# Ensemble with weighted averaging
print("Creating ensemble model...")
final_preds = 0.6 * lgb_val_preds + 0.4 * xgb_val_preds
ensemble_auc = roc_auc_score(y_val, final_preds)
print(f"Ensemble Validation AUC: {ensemble_auc:.4f}")

# Feature importance
def get_feature_importance(model, num_features=20):
    """Get feature importance from a tree-based model"""
    feature_names = (
        numerical_cols + 
        (preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist() 
         if categorical_cols else [])
    )
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance.head(num_features)

# Plot LightGBM feature importance
lgb_importance = get_feature_importance(lgb_model)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=lgb_importance)
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.savefig('plots/lgb_feature_importance.png')

# Plot XGBoost feature importance
xgb_importance = get_feature_importance(xgb_model)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('plots/xgb_feature_importance.png')

# Save models
print("Saving models...")
with open('models/lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Generate predictions for all users
print("\nGenerating predictions for all users...")
# Get all user IDs from submission sample
submission_users = submission_sample['user_id'].unique()
print(f"Total users for prediction: {len(submission_users)}")

# Create features for all users
all_users_features = all_features[all_features['user_id'].isin(submission_users)]

# Check if any users are missing
missing_users = set(submission_users) - set(all_users_features['user_id'])
if missing_users:
    print(f"Warning: {len(missing_users)} users from submission template are missing in our features")
    # For missing users, we'll use a default prediction (median of training predictions)

# Prepare data for prediction
X_all = all_users_features.drop(['user_id', 'first_transaction', 'last_transaction'], axis=1)
X_all_processed = preprocessor.transform(X_all)

# Generate predictions
lgb_all_preds = lgb_model.predict_proba(X_all_processed)[:, 1]
xgb_all_preds = xgb_model.predict_proba(X_all_processed)[:, 1]
final_all_preds = 0.6 * lgb_all_preds + 0.4 * xgb_all_preds

# Create submission dataframe for users we have predictions for
submission = pd.DataFrame({
    'user_id': all_users_features['user_id'],
    'pred_prob': final_all_preds
})

# Add any missing users with default prediction (median)
default_pred = np.median(final_all_preds)
if missing_users:
    missing_df = pd.DataFrame({
        'user_id': list(missing_users),
        'pred_prob': [default_pred] * len(missing_users)
    })
    submission = pd.concat([submission, missing_df])

# Ensure submission has the same user_ids as the sample
final_submission = pd.merge(
    submission_sample[['user_id']], 
    submission,
    on='user_id',
    how='left'
).fillna(default_pred)

# Save submission file
final_submission.to_csv('submissions/submission.csv', index=False)
print("Submission file created successfully!")

# Generate summary stats on predictions
print("\nPrediction summary:")
print(f"Min prediction: {final_submission['pred_prob'].min()}")
print(f"Max prediction: {final_submission['pred_prob'].max()}")
print(f"Mean prediction: {final_submission['pred_prob'].mean()}")
print(f"Predicted whales (>0.5): {(final_submission['pred_prob'] > 0.5).sum()} ({(final_submission['pred_prob'] > 0.5).mean()*100:.2f}%)")

# Plot prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(final_submission['pred_prob'], bins=50)
plt.title('Distribution of Whale Predictions')
plt.xlabel('Probability of Being a Whale')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/prediction_distribution.png')

print("\nAll done! Check the submissions directory for your final predictions.") 