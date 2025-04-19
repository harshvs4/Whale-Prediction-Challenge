import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
import os
from dotenv import load_dotenv
import json
import time
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create cache directory if it doesn't exist
CACHE_DIR = Path('../cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / 'llm_responses_cache.pkl'

# Load cache if exists
RESPONSE_CACHE = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE, 'rb') as f:
        RESPONSE_CACHE = pickle.load(f)

SYSTEM_PROMPT = """You are an expert financial analyst specializing in identifying high-value customers (whales) in cryptocurrency trading.
Given a user's transaction data and profile, provide ONLY a JSON response with these scores (0-1).
Be conservative with scores - most users are not whales.
If transaction volume or count is very low, all scores should be very low (< 0.1).

Required JSON format:
{
    "whale_potential_score": (score),
    "risk_tolerance_score": (score),
    "trading_sophistication_score": (score),
    "growth_trajectory_score": (score), 
    "wealth_indicator_score": (score)
}

Only return the JSON object, no other text."""

def get_cache_key(features):
    """Generate a cache key from user features"""
    # Round numerical values to reduce cache misses
    return f"{features['mean_gtv']:.0f}_{features['max_gtv']:.0f}_{features['total_gtv']:.0f}_{features['transaction_count']}_{features['buy_sell_ratio']:.1f}_{features['income_level']}_{features['age']:.0f}"

def extract_json_from_response(text):
    """Extract JSON from the response text"""
    try:
        return json.loads(text)
    except:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)
        if match:
            try:
                json_str = match.group()
                return json.loads(json_str)
            except:
                print(f"Failed to parse extracted JSON: {json_str}")
                return None
        return None

def get_llm_insights(user_data, retries=2):
    """Get insights from OpenAI API with caching and retries"""
    cache_key = get_cache_key(user_data)
    
    # Check cache first
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]
    
    for attempt in range(retries):
        try:
            data_str = f"""
            Transaction Stats:
            - Average Transaction Value: ${user_data['mean_gtv']:.2f}
            - Max Transaction: ${user_data['max_gtv']:.2f}
            - Total Volume: ${user_data['total_gtv']:.2f}
            - Transaction Count: {user_data['transaction_count']}
            - Buy/Sell Ratio: {user_data['buy_sell_ratio']:.2f}
            
            User Profile:
            - Income Level: {user_data['income_level']}
            - Device: {user_data['mobile_brand']} {user_data['mobile_model']}
            - Demographics: {user_data['gender']}, Age: {user_data['age']}, Education: {user_data['education']}
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 to reduce costs
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": data_str}
                ],
                temperature=0.1
            )
            
            scores = extract_json_from_response(response.choices[0].message.content)
            if scores:
                # Cache the successful response
                RESPONSE_CACHE[cache_key] = scores
                # Save cache periodically
                if len(RESPONSE_CACHE) % 5 == 0:  # Save more frequently
                    with open(CACHE_FILE, 'wb') as f:
                        pickle.dump(RESPONSE_CACHE, f)
                return scores
            else:
                print(f"Failed to parse response: {response.choices[0].message.content}")
                
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retry
            continue
            
    # If all retries failed, return default scores
    return {
        'whale_potential_score': 0.0,
        'risk_tolerance_score': 0.0,
        'trading_sophistication_score': 0.0,
        'growth_trajectory_score': 0.0,
        'wealth_indicator_score': 0.0
    }

def select_representative_samples(data, n_clusters=50):  # Reduced clusters to minimize API calls
    """Select representative samples using clustering"""
    # Select relevant numerical features for clustering
    features_for_clustering = ['mean_gtv', 'max_gtv', 'total_gtv', 'transaction_count', 'buy_sell_ratio', 'age']
    
    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features_for_clustering])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Select representative samples from each cluster
    representative_samples = []
    for i in range(n_clusters):
        cluster_samples = data[clusters == i]
        if len(cluster_samples) > 0:
            # Select the sample closest to cluster center
            cluster_center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(
                scaler.transform(cluster_samples[features_for_clustering]) - cluster_center, 
                axis=1
            )
            representative_idx = distances.argmin()
            representative_samples.append(cluster_samples.iloc[representative_idx])
    
    return pd.DataFrame(representative_samples)

def process_batch(batch_data):
    """Process a batch of data with LLM insights"""
    features_list = []
    for _, row in tqdm(batch_data.iterrows(), total=len(batch_data)):
        features = {
            'mean_gtv': float(row['mean_gtv']),
            'max_gtv': float(row['max_gtv']),
            'total_gtv': float(row['total_gtv']),
            'transaction_count': int(row['transaction_count']),
            'buy_sell_ratio': float(row['buy_sell_ratio']),
            'income_level': str(row['income_level']),
            'mobile_brand': str(row['mobile_brand']),
            'mobile_model': str(row['mobile_model']),
            'gender': str(row['gender']),
            'age': float(row['age']),
            'education': str(row['education'])
        }
        
        llm_scores = get_llm_insights(features)
        features.update(llm_scores)
        features_list.append(features)
        
        # Reduced sleep time since we have retries and error handling
        time.sleep(0.1)
    
    return features_list

def interpolate_scores(representative_samples, all_data, n_neighbors=5):
    """Interpolate scores for all samples based on representative samples"""
    # Features to use for finding nearest neighbors
    features_for_knn = ['mean_gtv', 'max_gtv', 'total_gtv', 'transaction_count', 'buy_sell_ratio', 'age']
    
    # Normalize features
    scaler = StandardScaler()
    rep_scaled = scaler.fit_transform(representative_samples[features_for_knn])
    all_scaled = scaler.transform(all_data[features_for_knn])
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(rep_scaled)
    distances, indices = nbrs.kneighbors(all_scaled)
    
    # Weight calculation
    weights = 1 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Score columns
    score_columns = [
        'whale_potential_score', 'risk_tolerance_score', 'trading_sophistication_score',
        'growth_trajectory_score', 'wealth_indicator_score'
    ]
    
    # Interpolate scores
    interpolated_scores = pd.DataFrame(index=all_data.index, columns=score_columns)
    for col in score_columns:
        rep_scores = representative_samples[col].values
        interpolated_scores[col] = np.sum(weights * rep_scores[indices], axis=1)
    
    return interpolated_scores

def main():
    print("Loading preprocessed data...")
    train_data = pd.read_csv('../data/preprocessed_train.csv')
    test_data = pd.read_csv('../data/preprocessed_test.csv')
    
    # Select representative samples
    print("Selecting representative samples...")
    n_clusters = 50  # Reduced clusters to minimize API calls
    train_representatives = select_representative_samples(train_data, n_clusters)
    test_representatives = select_representative_samples(test_data, n_clusters)
    
    # Process representative samples
    print("Processing representative samples...")
    train_rep_features = process_batch(train_representatives)
    test_rep_features = process_batch(test_representatives)
    
    # Convert to DataFrames
    train_rep_df = pd.DataFrame(train_rep_features)
    test_rep_df = pd.DataFrame(test_rep_features)
    
    # Interpolate scores for all samples
    print("Interpolating scores for all samples...")
    train_scores = interpolate_scores(train_rep_df, train_data)
    test_scores = interpolate_scores(test_rep_df, test_data)
    
    # Combine original features with interpolated scores
    train_final = pd.concat([train_data, train_scores], axis=1)
    test_final = pd.concat([test_data, test_scores], axis=1)
    
    # Save final features
    print("Saving LLM-enhanced features...")
    train_final.to_csv('../data/llm_train_features_v2.csv', index=False)
    test_final.to_csv('../data/llm_test_features_v2.csv', index=False)
    
    # Save final cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(RESPONSE_CACHE, f)
    
    print("LLM feature engineering completed!")

if __name__ == "__main__":
    main() 