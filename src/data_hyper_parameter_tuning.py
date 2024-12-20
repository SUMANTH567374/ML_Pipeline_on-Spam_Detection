import pickle
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from datetime import datetime

# Load the training and test data
train_data_path = r'.\data\processed\processed_train_data.csv'
test_data_path = r'.\data\processed\processed_test_data.csv'

try:
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    exit(1)

# Handle NaN values and ensure Cleaned_Message is a string
train_df['Cleaned_Message'] = train_df['Cleaned_Message'].fillna("").astype(str)
test_df['Cleaned_Message'] = test_df['Cleaned_Message'].fillna("").astype(str)

# Assuming 'Label' is the target and 'Cleaned_Message' is the feature for text classification
X_train = train_df['Cleaned_Message']
y_train = train_df['Label']
X_test = test_df['Cleaned_Message']
y_test = test_df['Label']

# Define the models and their parameter grids
models = {
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5],
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.1, 0.2],
    }),
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__penalty': ['l2'],
    })
}

# Iterate over the models, tuning and saving the best ones
for model_name, (model, param_grid) in models.items():
    print(f"\n=== Tuning {model_name} ===\n")
    
    # Define the pipeline
    pipeline = ImbPipeline([ 
        ('tfidf', TfidfVectorizer(max_features=1000, max_df=0.95, min_df=5, ngram_range=(1, 2))),  # TF-IDF Vectorizer
        ('oversample', SMOTE(random_state=42)),  # SMOTE Oversampling
        ('clf', model)  # Classifier
    ])
    
    # Set up StratifiedKFold cross-validation
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=stratified_cv,  # 5-fold stratified cross-validation
        scoring='f1',  # Optimize for F1-score
        verbose=2, 
        n_jobs=-1  # Use all available processors
    )
    
    try:
        # Fit the model using GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Best parameters and estimator
        print("Best Parameters:", grid_search.best_params_)
        print("Best Estimator:", grid_search.best_estimator_)
        
        # Save the best model
        model_save_path = f'./models/{model_name.lower().replace(" ", "_")}_best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Save the best model to a pickle file
        with open(model_save_path, 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        
        print(f"Best model for {model_name} saved to '{model_save_path}'")
        
        # Evaluate on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        print(f"\nClassification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"An error occurred while tuning or saving the model for {model_name}: {e}")
