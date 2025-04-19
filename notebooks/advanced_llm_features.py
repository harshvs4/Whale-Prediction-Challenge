import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Set up API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Enhanced prompts for better feature generation
OPENAI_PROMPT = """You are an expert financial analyst specializing in identifying high-value customers (whales). 
Analyze the following transaction data and provide numerical scores (0-1) for:
1. Transaction Value Score (based on min, max, mean values)
2. Transaction Pattern Score (based on consistency and frequency)
3. Wealth Indicator Score (based on device and user profile)
4. Risk Score (based on transaction volatility)
5. Growth Potential Score (based on transaction patterns)

Format your response as a JSON with these 5 scores."""

GEMINI_PROMPT = """As a financial expert, analyze this transaction data and provide numerical scores (0-1) for:
1. Transaction Value Score (based on min, max, mean values)
2. Transaction Pattern Score (based on consistency and frequency)
3. Wealth Indicator Score (based on device and user profile)
4. Risk Score (based on transaction volatility)
5. Growth Potential Score (based on transaction patterns)

Format your response as a JSON with these 5 scores."""

@lru_cache(maxsize=1000)
def get_openai_insights(transaction_data_str):
    """Get insights from OpenAI API with caching"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": OPENAI_PROMPT},
                {"role": "user", "content": transaction_data_str}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return None

@lru_cache(maxsize=1000)
def get_gemini_insights(transaction_data_str):
    """Get insights from Google Gemini API with caching"""
    try:
        response = model.generate_content(GEMINI_PROMPT + "\n" + transaction_data_str)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return None

def create_advanced_features(row):
    """Create advanced features using LLM insights"""
    # Convert data types
    try:
        age = float(row['age'])
        income_level = float(row['income_level'])
    except (ValueError, TypeError):
        age = 0
        income_level = 0
    
    # Prepare transaction data for analysis
    transaction_data = {
        'transaction_stats': {
            'min_gtv': float(row['min_gtv']),
            'max_gtv': float(row['max_gtv']),
            'mean_gtv': float(row['mean_gtv']),
            'std_gtv': float(row['std_gtv']),
            'total_gtv': float(row['total_gtv']),
            'transaction_count': int(row['transaction_count']),
            'buy_total_gtv': float(row['buy_total_gtv']),
            'sell_total_gtv': float(row['sell_total_gtv']),
            'buy_sell_ratio': float(row['buy_sell_ratio']),
            'transaction_frequency': float(row['transaction_frequency']),
            'transaction_span_days': int(row['transaction_span_days'])
        },
        'user_info': {
            'income_level': income_level,
            'mobile_brand': str(row['mobile_brand']),
            'mobile_model': str(row['mobile_model']),
            'gender': str(row['gender']),
            'education': str(row['education']),
            'age': age
        }
    }
    
    # Create features based on transaction patterns
    features = {}
    
    # Enhanced transaction pattern features
    features['transaction_value_volatility'] = float(row['std_gtv']) / float(row['mean_gtv']) if float(row['mean_gtv']) > 0 else 0
    features['transaction_size_growth'] = (float(row['max_gtv']) - float(row['min_gtv'])) / float(row['min_gtv']) if float(row['min_gtv']) > 0 else 0
    features['buy_sell_imbalance'] = abs(float(row['buy_total_gtv']) - float(row['sell_total_gtv'])) / float(row['total_gtv']) if float(row['total_gtv']) > 0 else 0
    
    # Enhanced user profile features
    features['age_income_interaction'] = age * income_level
    features['education_income_interaction'] = income_level
    
    # Enhanced device wealth indicators
    device_wealth_map = {
        'iPhone': 1.0,
        'Samsung': 0.8,
        'Xiaomi': 0.6,
        'OPPO': 0.5,
        'Vivo': 0.5,
        'Huawei': 0.7,
        'OnePlus': 0.7,
        'Realme': 0.4,
        'Nokia': 0.3,
        'Motorola': 0.4,
        'Sony': 0.8,
        'LG': 0.6,
        'Asus': 0.5,
        'Tecno': 0.3,
        'Infinix': 0.3,
        'Others': 0.5
    }
    
    mobile_brand = str(row['mobile_brand']).lower()
    features['device_wealth_score'] = device_wealth_map.get(mobile_brand, 0.5)
    
    # Enhanced transaction pattern indicators
    features['high_value_transaction_ratio'] = float(row['max_gtv']) / float(row['mean_gtv']) if float(row['mean_gtv']) > 0 else 0
    features['transaction_consistency'] = 1 - (float(row['std_gtv']) / float(row['mean_gtv'])) if float(row['mean_gtv']) > 0 else 0
    features['buying_power'] = float(row['buy_total_gtv']) / float(row['transaction_count']) if float(row['transaction_count']) > 0 else 0
    
    # New sophisticated features
    features['transaction_value_score'] = min(1.0, float(row['mean_gtv']) / 1000)  # Normalized to 0-1
    features['transaction_pattern_score'] = min(1.0, float(row['transaction_frequency']) / 10)  # Normalized to 0-1
    features['wealth_indicator_score'] = features['device_wealth_score'] * (1 + income_level/10)  # Combined device and income
    features['risk_score'] = features['transaction_value_volatility']  # Higher volatility = higher risk
    features['growth_potential_score'] = features['transaction_size_growth'] * features['transaction_consistency']
    
    # Additional interaction features
    features['value_pattern_interaction'] = features['transaction_value_score'] * features['transaction_pattern_score']
    features['wealth_risk_interaction'] = features['wealth_indicator_score'] * (1 - features['risk_score'])
    features['growth_consistency_interaction'] = features['growth_potential_score'] * features['transaction_consistency']
    
    return features

def process_batch(batch_data):
    """Process a batch of data"""
    features_list = []
    for _, row in batch_data.iterrows():
        features = create_advanced_features(row)
        features_list.append(features)
    return features_list

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_data = pd.read_csv('../data/preprocessed_train.csv')
    test_data = pd.read_csv('../data/preprocessed_test.csv')
    
    # Process data in batches
    batch_size = 1000
    train_features = []
    test_features = []
    
    print("Processing training data...")
    for i in tqdm(range(0, len(train_data), batch_size)):
        batch = train_data.iloc[i:i+batch_size]
        batch_features = process_batch(batch)
        train_features.extend(batch_features)
    
    print("Processing test data...")
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data.iloc[i:i+batch_size]
        batch_features = process_batch(batch)
        test_features.extend(batch_features)
    
    # Convert features to DataFrame
    train_features_df = pd.DataFrame(train_features)
    test_features_df = pd.DataFrame(test_features)
    
    # Save features
    print("Saving advanced features...")
    train_features_df.to_csv('../data/advanced_train_features.csv', index=False)
    test_features_df.to_csv('../data/advanced_test_features.csv', index=False)
    
    print("Advanced feature engineering completed!")

if __name__ == "__main__":
    main() 