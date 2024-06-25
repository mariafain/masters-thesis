import re
import string
from typing import List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils import get_x_y


def get_small_df(df: pd.DataFrame, target_class: str='generated', df_len: int=150000) -> pd.DataFrame:
    """
    This function returns a smaller size of the original dataset which contains an equal amount
    of rows for each target class value.   
    """
    df_pos = df[df[target_class] == 0]
    df_neg = df[df[target_class] == 1]

    df_small =  pd.concat([df_pos[:int(df_len/2)], df_neg[df_neg.shape[0] - int(df_len/2):]], ignore_index=True)
    
    return df_small.drop(['id'], axis=1)


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


def preprocess_df(df: pd.DataFrame, text_column: str='text', preprocessed_col_name: str='preprocessed_text') -> pd.DataFrame:
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

def prepare_data_for_predicting(data: pd.DataFrame):
    """
    Prepares a pandas DataFrame for model prediction by preprocessing the texts and dividing the dataset to X and y. 
    """
    data = preprocess_df(data)
    x, y = get_x_y(data)

    return x, y
