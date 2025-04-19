import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# OpenAI API key (you'll need to set this in your environment)
# import openai
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# For demo purposes, we'll use a simulated LLM response
# In practice, you would use a real LLM API like OpenAI, HuggingFace, etc.

class LLMFeatureEngineer:
    """Class for using LLM to enhance feature engineering for the whale prediction task"""
    
    def __init__(self, profile_data, transaction_data, api_key=None):
        self.profile_data = profile_data
        self.transaction_data = transaction_data
        self.api_key = api_key
        
    def extract_device_embeddings(self, save_path=None):
        """Extract embeddings from mobile device information using simulated LLM"""
        print("Extracting device embeddings from mobile brand and marketing name...")
        
        # Combine brand and marketing name
        self.profile_data['device_info'] = self.profile_data['mobile_brand_name'] + ' ' + \
                                           self.profile_data['mobile_marketing_name']
        
        # Use TF-IDF and SVD to create embeddings (simulating what an LLM might do)
        # In a real implementation, you would send this text to an LLM API
        vectorizer = TfidfVectorizer(min_df=2, max_features=100)
        device_matrix = vectorizer.fit_transform(self.profile_data['device_info'].fillna(''))
        
        # Reduce to lower dimensions
        svd = TruncatedSVD(n_components=5, random_state=42)
        device_embeddings = svd.fit_transform(device_matrix)
        
        # Add embeddings as features
        for i in range(device_embeddings.shape[1]):
            self.profile_data[f'device_emb_{i}'] = device_embeddings[:, i]
            
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump((vectorizer, svd), f)
                
        print(f"Added {device_embeddings.shape[1]} device embedding features")
        return self.profile_data
    
    def categorize_transaction_patterns(self, user_trx_data):
        """
        Simulate LLM analysis of transaction patterns
        In a real implementation, you would send transaction data to an LLM API
        and get back classifications or embeddings
        """
        patterns = {}
        
        for user_id, user_df in user_trx_data.groupby('user_id'):
            # Get transaction counts by hour
            hour_counts = user_df['transaction_time'].dt.hour.value_counts()
            
            # Determine if user is morning/afternoon/evening/night trader
            max_hour = hour_counts.idxmax() if not hour_counts.empty else 0
            
            if 5 <= max_hour < 12:
                time_pattern = "morning_trader"
            elif 12 <= max_hour < 17:
                time_pattern = "afternoon_trader"
            elif 17 <= max_hour < 22:
                time_pattern = "evening_trader"
            else:
                time_pattern = "night_trader"
                
            # Look at buy vs sell behavior
            buy_count = user_df[user_df['transaction_type'] == 'BUY'].shape[0]
            sell_count = user_df[user_df['transaction_type'] == 'SELL'].shape[0]
            
            if buy_count > 2*sell_count:
                trade_pattern = "heavy_buyer"
            elif sell_count > 2*buy_count:
                trade_pattern = "heavy_seller"
            else:
                trade_pattern = "balanced_trader"
                
            # Look at consistency in transaction amounts
            gtv_std = user_df['gtv'].std() if user_df.shape[0] > 1 else 0
            gtv_mean = user_df['gtv'].mean() if user_df.shape[0] > 0 else 0
            
            if gtv_std < 0.2 * gtv_mean and gtv_mean > 0:
                amount_pattern = "consistent_amount"
            elif gtv_std > gtv_mean and gtv_mean > 0:
                amount_pattern = "variable_amount"
            else:
                amount_pattern = "neutral_amount"
                
            patterns[user_id] = {
                "time_pattern": time_pattern,
                "trade_pattern": trade_pattern,
                "amount_pattern": amount_pattern
            }
            
        # Convert to DataFrame
        pattern_df = pd.DataFrame.from_dict(patterns, orient='index')
        pattern_df.index.name = 'user_id'
        pattern_df.reset_index(inplace=True)
        
        # One-hot encode the patterns
        pattern_dummies = pd.get_dummies(pattern_df, columns=['time_pattern', 'trade_pattern', 'amount_pattern'])
        
        return pattern_dummies
    
    def analyze_transaction_sequences(self, min_transactions=5):
        """
        Analyze the sequence of transactions to identify patterns
        This simulates what an LLM might do to understand sequential behaviors
        """
        print("Analyzing transaction sequences...")
        
        # Sort transactions by time for each user
        sorted_trx = self.transaction_data.sort_values(['user_id', 'transaction_time'])
        
        # Create a "sequence" column representing transaction type and approximate amount
        def amount_bucket(x):
            if x < 10:
                return "S"  # Small
            elif x < 100:
                return "M"  # Medium
            else:
                return "L"  # Large
                
        sorted_trx['amount_bucket'] = sorted_trx['gtv'].apply(amount_bucket)
        sorted_trx['trx_sequence'] = sorted_trx['transaction_type'] + "_" + sorted_trx['amount_bucket']
        
        # Group by user and collect sequences
        user_sequences = {}
        for user_id, user_df in sorted_trx.groupby('user_id'):
            if user_df.shape[0] >= min_transactions:
                # Take the first 10 transactions (or all if fewer)
                sequence = user_df['trx_sequence'].values[:10].tolist()
                user_sequences[user_id] = sequence
        
        # Analyze common patterns
        # We'll look for common sub-sequences
        def contains_pattern(seq, pattern):
            if len(pattern) > len(seq):
                return False
            for i in range(len(seq) - len(pattern) + 1):
                if seq[i:i+len(pattern)] == pattern:
                    return True
            return False
        
        # Define some interesting patterns we might find
        patterns = {
            "buy_small_then_sell_large": ["BUY_S", "SELL_L"],
            "buy_large_then_sell_small": ["BUY_L", "SELL_S"],
            "buy_medium_sequence": ["BUY_M", "BUY_M", "BUY_M"],
            "sell_medium_sequence": ["SELL_M", "SELL_M", "SELL_M"],
            "alternating_buy_sell": ["BUY_M", "SELL_M", "BUY_M", "SELL_M"]
        }
        
        # Check each user for each pattern
        pattern_results = {user_id: {p: contains_pattern(seq, patterns[p]) for p in patterns} 
                          for user_id, seq in user_sequences.items()}
        
        # Convert to DataFrame
        pattern_df = pd.DataFrame.from_dict(pattern_results, orient='index')
        pattern_df.index.name = 'user_id'
        pattern_df = pattern_df.reset_index()
        
        # Convert boolean to int
        for p in patterns:
            pattern_df[p] = pattern_df[p].astype(int)
            
        return pattern_df
    
    def device_wealth_indicator(self):
        """
        Use LLM knowledge to categorize mobile devices by approximate price/premium level
        This simulates LLM knowledge about device market positioning
        """
        print("Generating wealth indicators from device information...")
        
        # Define premium brands and models
        premium_brands = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Huawei']
        budget_brands = ['Xiaomi', 'Redmi', 'POCO', 'Realme', 'Nokia', 'Oppo', 'Vivo']
        
        # Define premium model keywords
        premium_keywords = ['Pro', 'Max', 'Ultra', 'Note', 'Plus', 'Edge']
        
        # Calculate a wealth score based on brand and model
        def wealth_score(row):
            brand = row['mobile_brand_name']
            model = str(row['mobile_marketing_name'])
            
            base_score = 5  # middle score
            
            # Adjust for brand
            if brand in premium_brands:
                base_score += 2
            elif brand in budget_brands:
                base_score -= 2
                
            # Adjust for model keywords
            if any(keyword in model for keyword in premium_keywords):
                base_score += 1
                
            # Adjust for specific high-end indicators
            if brand == 'Apple' and ('Pro' in model or 'Max' in model):
                base_score += 1
            if brand == 'Samsung' and ('S2' in model or 'Note' in model or 'Fold' in model):
                base_score += 1
                
            return min(max(base_score, 1), 10)  # Keep score between 1-10
            
        self.profile_data['wealth_indicator'] = self.profile_data.apply(wealth_score, axis=1)
        
        # Create wealth level categories
        self.profile_data['wealth_level'] = pd.cut(
            self.profile_data['wealth_indicator'], 
            bins=[0, 3, 6, 10], 
            labels=['budget', 'mid_range', 'premium']
        )
        
        # One-hot encode wealth level
        wealth_dummies = pd.get_dummies(self.profile_data[['user_id', 'wealth_level']], 
                                       columns=['wealth_level'], 
                                       prefix='wealth')
        
        return wealth_dummies
    
    def generate_all_llm_features(self):
        """Generate all LLM-based features and return combined DataFrame"""
        print("Generating all LLM-enhanced features...")
        
        # Device embeddings
        self.extract_device_embeddings()
        
        # Transaction patterns
        pattern_features = self.categorize_transaction_patterns(self.transaction_data)
        
        # Sequence analysis
        sequence_features = self.analyze_transaction_sequences()
        
        # Wealth indicators
        wealth_features = self.device_wealth_indicator()
        
        # Combine all features into one dataframe
        # Start with profile data (already has device embeddings)
        llm_features = self.profile_data[['user_id'] + [col for col in self.profile_data.columns 
                                                      if col.startswith('device_emb_')]]
        
        # Merge with other feature sets
        feature_dfs = [pattern_features, sequence_features, wealth_features]
        for df in feature_dfs:
            if not df.empty:
                llm_features = pd.merge(llm_features, df, on='user_id', how='left')
        
        # Fill NAs
        llm_features = llm_features.fillna(0)
        
        print(f"Generated {llm_features.shape[1]-1} LLM-enhanced features for {llm_features.shape[0]} users")
        return llm_features

# The following section would be executed when running the script
if __name__ == "__main__":
    print("Loading data...")
    profile_data = pd.read_csv('../data/profile.csv')
    transaction_data = pd.read_csv('../data/trx_data.csv')
    transaction_data['transaction_time'] = pd.to_datetime(transaction_data['transaction_time'])
    
    print("Initializing LLM Feature Engineer...")
    feature_engineer = LLMFeatureEngineer(profile_data, transaction_data)
    
    print("Generating LLM-enhanced features...")
    llm_features = feature_engineer.generate_all_llm_features()
    
    print("Saving LLM-enhanced features...")
    llm_features.to_csv('llm_features.csv', index=False)
    
    print("Feature generation complete!")
    print(f"Generated features: {llm_features.columns.tolist()}")
    
    # Visualize some of the LLM-generated features
    plt.figure(figsize=(10, 6))
    sns.countplot(data=llm_features, x='wealth_level')
    plt.title('Distribution of Device Wealth Levels')
    plt.savefig('wealth_distribution.png')
    
    if 'time_pattern_morning_trader' in llm_features.columns:
        patterns = ['time_pattern_morning_trader', 'time_pattern_afternoon_trader', 
                   'time_pattern_evening_trader', 'time_pattern_night_trader']
        
        pattern_counts = llm_features[patterns].sum().reset_index()
        pattern_counts.columns = ['Pattern', 'Count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pattern_counts, x='Pattern', y='Count')
        plt.title('Distribution of Trading Time Patterns')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('time_patterns.png')
    
    print("Analysis complete!") 