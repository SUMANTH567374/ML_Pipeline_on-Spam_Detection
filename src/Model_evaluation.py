import pickle
import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime

test_data_path = r'.\data\processed\processed_test_data.csv'
test_df = pd.read_csv(test_data_path)

# Check for missing values in the dataset
print(f"Missing values in the dataset:\n{test_df.isnull().sum()}")

# Handle missing values: Drop rows with missing target or features
test_df.dropna(subset=['Cleaned_Message', 'Label'], inplace=True)

# Assuming 'Label' is the target and 'Cleaned_Message' is the feature for text classification
X_test = test_df['Cleaned_Message']
y_test = test_df['Label']

# Check if the data is loaded correctly
print(f"Test data loaded. Number of records: {len(test_df)}")

# Define models and the corresponding paths where the best models were saved
model_paths = {
    'Random Forest': r'./models\random_forest_best_model_20241220_223535.pkl',
    'XGBoost': r'./models\xgboost_best_model_20241220_223635.pkl',
    'Logistic Regression': r'./models\logistic_regression_best_model_20241220_223638.pkl'
}

# Create a dictionary to store the metrics for each model
model_metrics = {}

# Load each model, predict on the test set, and calculate the metrics
for model_name, model_path in model_paths.items():
    try:
        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Model file for {model_name} not found at '{model_path}'. Skipping...")
            continue
        
        # Load the model from pickle file (the model includes the vectorizer and classifier)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if model is loaded correctly
        print(f"{model_name} model loaded successfully.")
        
        # Make predictions (Ensure X_test is the right format)
        print(f"Making predictions for {model_name}...")
        y_pred = model.predict(X_test)
        
        # Check if predictions are made
        if len(y_pred) == 0:
            print(f"No predictions were made for {model_name}.")
            continue
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save the metrics for the model
        model_metrics[model_name] = report

        print(f"Metrics for {model_name}:")
        print(report)

    except Exception as e:
        print(f"An error occurred while loading or evaluating the model {model_name}: {e}")

# Check if there are any metrics to save
if model_metrics:
    # Create the 'metrics' directory if it doesn't exist
    metrics_dir = './metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    # Save the metrics in a JSON file within the 'metrics' directory
    metrics_save_path = os.path.join(metrics_dir, 'model_metrics.json')

    try:
        with open(metrics_save_path, 'w') as f:
            json.dump(model_metrics, f, indent=4)
        print(f"Model metrics saved to '{metrics_save_path}'")
    except Exception as e:
        print(f"An error occurred while saving the metrics: {e}")
else:
    print("No metrics were generated.")


