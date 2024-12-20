import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import re
import spacy
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load NLTK stopwords
nltk_stopwords = set(stopwords.words('english'))

# Step 1: Preprocess Text Data
def preprocess_text(text):
    """
    Clean and preprocess the message text for spam classification.
    """
    # 1. Lowercase the text
    text = text.lower()   

    # 2. Replace URLs and emails with placeholders
    text = re.sub(r"http\S+|www\S+|https\S+", "url_placeholder", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "email_placeholder", text)
    
    # 3. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # 4. Tokenize using spaCy
    doc = nlp(text)
    
    # 5. Lemmatize and remove stop words
    tokens = [
        token.lemma_ for token in doc 
        if token.is_alpha and token.text not in nltk_stopwords
    ]
    
    # 6. Join tokens back into a single string
    cleaned_text = " ".join(tokens)
    
    return cleaned_text

# Step 2: Process Data
def process_data(data, drop_columns=None):
    """
    Processes the dataset for spam classification.
    - Drops unwanted columns.
    - Renames columns.
    - Cleans and preprocesses text.
    """
    # Drop rows with missing values or handle missing values appropriately
    data.dropna(subset=['v1', 'v2'], inplace=True)  # Ensures that essential columns are not missing

    # Drop unwanted columns if provided
    if drop_columns:
        data.drop(drop_columns, axis=1, inplace=True)

    # Rename columns as per the request
    data = data.rename(columns={'v1': 'Label', 'v2': 'Message'})
    
    # Clean and preprocess text
    data['Cleaned_Message'] = data['Message'].apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])

    return data

# Step 3: Process and Save Data
def process_and_save(data_path, output_dir, drop_columns=None, output_filename="processed_spam_data.csv"):
    """
    Processes the dataset and saves the processed version.
    """
    # Load dataset
    data = pd.read_csv(data_path)

    # Process dataset
    processed_data = process_data(data, drop_columns)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    output_path = os.path.join(output_dir, output_filename)
    processed_data.to_csv(output_path, index=False)

    print(f"Data processed and saved to: {output_path}")

    return processed_data

# Define paths for train and test datasets
train_path = './data/raw/train.csv'  # Train dataset path
test_path = './data/raw/test.csv'    # Test dataset path
output_dir = './data/processed'

# Define columns to drop (if any)
drop_columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

# Process and save train dataset
processed_train_data = process_and_save(train_path, output_dir, drop_columns, output_filename="processed_train_data.csv")

# Process and save test dataset
processed_test_data = process_and_save(test_path, output_dir, drop_columns, output_filename="processed_test_data.csv")



