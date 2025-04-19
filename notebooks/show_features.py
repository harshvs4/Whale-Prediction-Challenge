import pickle
import pandas as pd
import numpy as np

# Load the model and preprocessor
with open('models/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Get feature names
numerical_cols = preprocessor.transformers_[0][2]
categorical_cols = preprocessor.transformers_[1][2]

# This is the part that might need adjustment depending on the actual structure
feature_names = list(numerical_cols)
if len(categorical_cols) > 0:
    try:
        # Try to get one-hot encoded feature names
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
        feature_names += cat_feature_names
    except:
        # Fallback if structure is different
        print("Couldn't get categorical feature names, using indices only")

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': range(len(model.feature_importances_)),  # Use indices as fallback
    'Importance': model.feature_importances_
})

# If we have feature names, update the Feature column
if len(feature_names) == len(model.feature_importances_):
    feature_importance['Feature'] = feature_names

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Print top 20 features
print("Top 20 features by importance:")
print(feature_importance.head(20))

# Also show the numerical features by index for reference
print("\nNumerical feature mapping:")
for i, col in enumerate(numerical_cols):
    print(f"Index {i}: {col}")

# Save to CSV for easier viewing
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nFull feature importance saved to feature_importance.csv") 