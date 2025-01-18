import os
import pandas as pd
from preprocessing import preprocess_text
from vectorization import vectorize_text

def load_and_preprocess_dataset(data_path, text_column):
    """
    Load the dataset, validate the text column, and preprocess the text data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    if text_column not in df.columns:
        raise KeyError(f"The dataset does not contain a '{text_column}' column.")

    df[text_column] = df[text_column].fillna("").apply(preprocess_text)
    return df

if __name__ == "__main__":
    DATA_PATH = "data/fakeReviewData.csv"
    TEXT_COLUMN = "text"
    OUTPUT_PATH = "output/FakeReviewDataPreprocessed.csv"
    MAX_FEATURES = 5000

    try:
        df = load_and_preprocess_dataset(DATA_PATH, TEXT_COLUMN)
        tfidf_df, _ = vectorize_text(df, TEXT_COLUMN, max_features=MAX_FEATURES)
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
