import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load and merge all data files"""
    print("Loading data files...")
    
    # Load transaction data
    trx_data = pd.read_csv('../data/trx_data.csv')
    
    # Load profile data
    profile = pd.read_csv('../data/profile.csv')
    
    # Load training labels
    train_labels = pd.read_csv('../data/train_label.csv')
    
    # Load submission sample
    submission_sample = pd.read_csv('../data/submission_sample.csv')
    
    return trx_data, profile, train_labels, submission_sample

def create_features(trx_data, profile):
    """Create features from transaction and profile data"""
    print("Creating features...")
    
    # Create buy and sell GTV columns
    trx_data['buy_gtv'] = trx_data.apply(lambda x: x['gtv'] if x['transaction_type'] == 'BUY' else 0, axis=1)
    trx_data['sell_gtv'] = trx_data.apply(lambda x: x['gtv'] if x['transaction_type'] == 'SELL' else 0, axis=1)
    
    # Group transaction data by user_id
    user_trx = trx_data.groupby('user_id').agg({
        'gtv': ['min', 'max', 'mean', 'std', 'sum', 'count'],
        'buy_gtv': ['sum'],
        'sell_gtv': ['sum'],
        'transaction_time': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    user_trx.columns = ['user_id', 'min_gtv', 'max_gtv', 'mean_gtv', 'std_gtv', 
                       'total_gtv', 'transaction_count', 'buy_total_gtv', 
                       'sell_total_gtv', 'first_transaction', 'last_transaction']
    
    # Calculate additional features
    user_trx['buy_sell_ratio'] = user_trx['buy_total_gtv'] / (user_trx['sell_total_gtv'] + 1e-6)
    user_trx['transaction_frequency'] = user_trx['transaction_count'] / (
        (pd.to_datetime(user_trx['last_transaction']) - 
         pd.to_datetime(user_trx['first_transaction'])).dt.days + 1
    )
    user_trx['transaction_span_days'] = (
        pd.to_datetime(user_trx['last_transaction']) - 
        pd.to_datetime(user_trx['first_transaction'])
    ).dt.days
    
    # Merge with profile data
    df = pd.merge(user_trx, profile, on='user_id', how='left')
    
    # Encode categorical variables
    le = LabelEncoder()
    df['mobile_brand'] = le.fit_transform(df['mobile_brand_name'].fillna('unknown'))
    df['mobile_model'] = le.fit_transform(df['mobile_marketing_name'].fillna('unknown'))
    df['gender'] = le.fit_transform(df['gender_name'].fillna('unknown'))
    df['marital_status'] = le.fit_transform(df['marital_status'].fillna('unknown'))
    df['education'] = le.fit_transform(df['education_background'].fillna('unknown'))
    df['occupation'] = le.fit_transform(df['occupation'].fillna('unknown'))
    
    # Create age features
    df['age'] = df['age_in_year'].fillna(df['age_in_year'].mean())
    df['age_squared'] = df['age'] ** 2
    
    # Drop unnecessary columns
    df = df.drop(['first_transaction', 'last_transaction', 
                  'mobile_brand_name', 'mobile_marketing_name',
                  'gender_name', 'marital_status', 'education_background',
                  'occupation', 'age_in_year'], axis=1)
    
    return df

def main():
    # Load data
    trx_data, profile, train_labels, submission_sample = load_data()
    
    # Create features
    df = create_features(trx_data, profile)
    
    # Split into train and test sets
    train_data = pd.merge(df, train_labels, on='user_id', how='inner')
    test_data = pd.merge(df, submission_sample, on='user_id', how='inner')
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    train_data.to_csv('../data/preprocessed_train.csv', index=False)
    test_data.to_csv('../data/preprocessed_test.csv', index=False)
    
    print("Preprocessing complete!")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

if __name__ == "__main__":
    main() 