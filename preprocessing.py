import re
import string
from typing import List

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def clean_text(input_text: str) -> List[str]:
    """
    This function preprocesses the input text string. The preprocessing includes
    converting the text to lowercase, removing links, mentions, punctuation and stopwords.
    Additionally, multiple spaces are replaced with single spaces and words are lemmatized.
    The text is tokenized and the function returns a list of preprocessed tokens from the 
    original input text.
    """
    text = input_text.lower()

    # Remove links
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)
    # Remove @mentions
    text = re.sub(r"\s@\S+", " ", text)
    # Remove all punctuation
    punctuation_table = str.maketrans("", "", string.punctuation)
    text = text.translate(punctuation_table)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def preprocess_df(df: pd.DataFrame, text_column: str, preprocessed_col_name: str='preprocessed_text') -> pd.DataFrame:
    """
    This function preprocesses all texts from the column `text_column` of the dataframe `df` with the
    `clean_text` function. It returns the dataframe with a new column `preprocessed_col_name` containing
    preprocessed texts. Note that the texts in the new column are not tokenized.
    """
    preprocessed_texts = []
    for text in df[text_column]:
        preprocessed_texts.append(clean_text(text))
    
    df[preprocessed_col_name] = [' '.join(text) for text in preprocessed_texts]
    return df
