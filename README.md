# Whale Prediction Challenge

This project is focused on predicting whale behavior using machine learning techniques. The project includes various Python scripts for data preprocessing, feature engineering, model training, and visualization.

## Project Structure

```
.
├── data/                    # Data files (not tracked in Git)
│   ├── trx_data.csv        # Transaction data
│   ├── profile.csv         # Profile data
│   ├── train_label.csv     # Training labels
│   ├── submission_sample.csv # Sample submission format
│   └── preprocessed_*.csv  # Preprocessed data files
│
├── notebooks/              # Python scripts
│   ├── preprocess_data.py  # Data preprocessing
│   ├── whale_prediction.py # Basic prediction model
│   ├── whale_prediction_advanced.py # Advanced prediction model
│   ├── whale_prediction_with_llm.py # LLM-based prediction
│   ├── generate_submissions*.py # Submission generation scripts
│   ├── generate_visualizations.py # Visualization generation
│   └── show_features.py    # Feature analysis
│
├── submissions/            # Generated submission files
├── cache/                  # Cache directory
└── venv/                   # Python virtual environment
```

## Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/harshvs4/Whale-Prediction-Challenge.git
cd Whale-Prediction-Challenge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. Download and place the data files:
   - Due to file size limitations, data files are not included in the repository
   - Place the following files in the `data/` directory:
     - trx_data.csv
     - profile.csv
     - train_label.csv
     - submission_sample.csv

## Data Files

- `trx_data.csv`: Contains transaction data
- `profile.csv`: Contains profile information
- `train_label.csv`: Contains training labels
- `submission_sample.csv`: Sample submission format

## Scripts Overview

### Data Preprocessing
- `preprocess_data.py`: Handles data cleaning and preprocessing
- `show_features.py`: Analyzes and displays feature information

### Prediction Models
- `whale_prediction.py`: Basic prediction model
- `whale_prediction_advanced.py`: Advanced prediction model with additional features
- `whale_prediction_with_llm.py`: LLM-based prediction model

### Feature Engineering
- `llm_feature_engineering.py`: Feature engineering using LLM
- `advanced_llm_features.py`: Advanced LLM-based features
- `advanced_llm_features_v2.py`: Updated version of advanced LLM features

### Submission Generation
- `generate_submissions.py`: Generates submission files
- `generate_submissions_v2.py`: Updated version of submission generation
- `generate_submissions_llm.py`: LLM-based submission generation

### Visualization
- `generate_visualizations.py`: Creates visualizations of the data and results

## Usage

1. First, preprocess the data:
```bash
python notebooks/preprocess_data.py
```

2. Run the desired prediction model:
```bash
python notebooks/whale_prediction.py
# or
python notebooks/whale_prediction_advanced.py
# or
python notebooks/whale_prediction_with_llm.py
```

3. Generate submissions:
```bash
python notebooks/generate_submissions.py
```

4. Generate visualizations:
```bash
python notebooks/generate_visualizations.py
```

## Output

- Generated submission files will be saved in the `submissions/` directory
- Visualizations will be saved in the `notebooks/plots/` directory
- Preprocessed data will be saved in the `data/` directory

## Notes

- Make sure to have sufficient disk space as the data files are large
- The project uses both traditional machine learning and LLM-based approaches
- Different versions of scripts (v2) indicate improved or updated implementations
- Data files are not tracked in Git due to size limitations. Please obtain them separately and place them in the `data/` directory
- Keep your `.env` file secure and never commit it to version control
