import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(df, text_column, max_features=500):
    """
    Vectorize text data using TF-IDF.

    Parameters:
    - df (DataFrame): DataFrame containing text data.
    - text_column (str): Name of the column containing preprocessed text.
    - max_features (int): Maximum number of features for TF-IDF.

    Returns:
    - tfidf_df (DataFrame): TF-IDF features as a DataFrame.
    - tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english'
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )
    return tfidf_df, tfidf_vectorizer
