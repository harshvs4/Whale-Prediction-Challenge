import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create plots directory if it doesn't exist
plots_dir = Path('notebooks/plots')
plots_dir.mkdir(exist_ok=True)

def load_data():
    """Load the necessary data files"""
    data_dir = Path('data')
    train = pd.read_csv(data_dir / 'preprocessed_train.csv')
    test = pd.read_csv(data_dir / 'preprocessed_test.csv')
    llm_train = pd.read_csv(data_dir / 'llm_train_features_v2.csv')
    llm_test = pd.read_csv(data_dir / 'llm_test_features_v2.csv')
    return train, test, llm_train, llm_test

def plot_transaction_distribution(train):
    """Plot distribution of transaction amounts"""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=train, x='total_gtv', bins=50, log_scale=True)
    plt.title('Distribution of Total Transaction Value (Log Scale)')
    plt.xlabel('Total GTV (Log Scale)')
    plt.ylabel('Count')
    plt.savefig(plots_dir / 'transaction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_demographics(train):
    """Plot user demographic distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Age distribution
    sns.histplot(data=train, x='age', bins=30, ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')
    
    # Education level
    sns.countplot(data=train, x='education', ax=axes[0, 1])
    axes[0, 1].set_title('Education Level Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Gender distribution
    sns.countplot(data=train, x='gender', ax=axes[1, 0])
    axes[1, 0].set_title('Gender Distribution')
    
    # Device brand distribution
    sns.countplot(data=train, x='mobile_brand', ax=axes[1, 1])
    axes[1, 1].set_title('Device Brand Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'user_demographics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_correlations(train):
    """Plot correlation matrix of important features"""
    # Select numerical features
    numerical_features = ['min_gtv', 'max_gtv', 'mean_gtv', 'std_gtv', 'total_gtv', 
                         'transaction_count', 'buy_total_gtv', 'sell_total_gtv', 
                         'buy_sell_ratio', 'transaction_frequency', 'age']
    
    corr_matrix = train[numerical_features].corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_llm_feature_distributions(llm_train):
    """Plot distributions of LLM-generated features"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['whale_potential_score', 'risk_tolerance_score', 
               'trading_sophistication_score', 'growth_trajectory_score',
               'wealth_indicator_score']
    
    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        sns.histplot(data=llm_train, x=feature, bins=30, ax=axes[row, col])
        axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'llm_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_comparison():
    """Plot comparison of model performance metrics"""
    models = ['XGBoost', 'LightGBM', 'Ensemble', 'XGBoost+LLM', 'LightGBM+LLM', 'Ensemble+LLM']
    metrics = {
        'AUC': [0.8665, 0.8672, 0.8701, 0.8620, 0.8662, 0.8641],
        'Precision': [0.782, 0.785, 0.791, 0.778, 0.784, 0.786],
        'Recall': [0.745, 0.748, 0.752, 0.742, 0.747, 0.749],
        'F1-Score': [0.763, 0.766, 0.771, 0.759, 0.765, 0.767]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (metric, values) in enumerate(metrics.items()):
        row = i // 2
        col = i % 2
        sns.barplot(x=models, y=values, ax=axes[row, col])
        axes[row, col].set_title(f'{metric} Comparison')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].set_ylim(min(values) - 0.02, max(values) + 0.02)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance_comparison():
    """Plot comparison of feature importance between models"""
    features = ['min_gtv', 'buy_total_gtv', 'std_gtv', 'max_gtv', 'mean_gtv',
               'buy_sell_ratio', 'transaction_frequency']
    importance_scores = {
        'XGBoost': [0.156, 0.142, 0.128, 0.115, 0.098, 0.089, 0.075],
        'LightGBM': [0.152, 0.138, 0.125, 0.112, 0.095, 0.085, 0.072],
        'Ensemble': [0.158, 0.145, 0.130, 0.118, 0.100, 0.092, 0.078]
    }
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(features))
    width = 0.25
    
    for i, (model, scores) in enumerate(importance_scores.items()):
        plt.bar(x + i*width, scores, width, label=model)
    
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Comparison Across Models')
    plt.xticks(x + width, features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transaction_patterns(train):
    """Plot transaction patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Transaction count distribution
    sns.histplot(data=train, x='transaction_count', bins=30, ax=axes[0, 0])
    axes[0, 0].set_title('Transaction Count Distribution')
    
    # Buy/Sell ratio distribution
    sns.histplot(data=train, x='buy_sell_ratio', bins=30, ax=axes[0, 1])
    axes[0, 1].set_title('Buy/Sell Ratio Distribution')
    axes[0, 1].set_xscale('log')
    
    # Transaction frequency distribution
    sns.histplot(data=train, x='transaction_frequency', bins=30, ax=axes[1, 0])
    axes[1, 0].set_title('Transaction Frequency Distribution')
    
    # Transaction span days distribution
    sns.histplot(data=train, x='transaction_span_days', bins=30, ax=axes[1, 1])
    axes[1, 1].set_title('Transaction Span Days Distribution')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'transaction_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    train, test, llm_train, llm_test = load_data()
    
    print("Generating visualizations...")
    plot_transaction_distribution(train)
    plot_user_demographics(train)
    plot_feature_correlations(train)
    plot_llm_feature_distributions(llm_train)
    plot_model_performance_comparison()
    plot_feature_importance_comparison()
    plot_transaction_patterns(train)
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main() 