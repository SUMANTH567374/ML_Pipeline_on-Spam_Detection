import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(data_url: str) -> pd.DataFrame:
    try:
        # Load the CSV with 'latin1' encoding
        df = pd.read_csv(data_url, encoding='latin1')
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {data_url}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        print(e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # Specify the folder where you want to save the data (e.g., 'data/raw')
        data_path = os.path.join(data_path, 'raw')
        
        # Create the folder if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Save the train and test datasets
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main():
    try:
        # Path to the local spam.csv file
        data_path = r"C:\Users\LENOVO\Downloads\spam.csv"
        df = load_data(data_url=data_path)

        # Split the data into training and testing sets (80% train, 20% test)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        # Save the split data to a separate folder (e.g., 'data/raw')
        save_data(train_data, test_data, data_path='data')

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")

if __name__ == '__main__':
    main()

