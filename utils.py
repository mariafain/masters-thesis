import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

MAX_SEQ_LEN = 128


def init_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

def get_x_y(df: pd.DataFrame, feature: str='preprocessed_text', target_class: str='generated'):
    x = df[feature] #.values
    y = np.array(df[target_class])
    
    return x, y

def stratified_split(df: pd.DataFrame, target_class: str='generated') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function splits the given dataframe to a training, validation and testing set using the stratified 
    shuffle split. It returns the training set, validation set and the testing set, respectively.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

    for train_index, test_valid_index in split.split(df, df[target_class]):
        train_set = df.iloc[train_index]
        test_valid_set = df.iloc[test_valid_index]

    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for test_index, valid_index in split2.split(test_valid_set, test_valid_set[target_class]):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    return train_set, valid_set, test_set

def load_extra_data(path_to_extra_data):
    """
    Loads extra datasets needed for validation. The argument `path_to_extra_data` is the directory where the extra data is stored.
    Returns a concatenated pandas Dataframe with the three extra datasets.
    """
    df_palm = pd.read_csv(os.path.join(os.getcwd(), path_to_extra_data, 'LLM_generated_essay_PaLM.csv'), header=0)
    df_falcon = pd.read_csv(os.path.join(os.getcwd(), path_to_extra_data, 'falcon_180b_v1.csv'), header=0)
    df_llama = pd.read_csv(os.path.join(os.getcwd(), path_to_extra_data, 'llama_70b_v1.csv'), header=0)

    df_llama.rename(columns={'generated_text': 'text'}, inplace=True)
    df_falcon.rename(columns={'generated_text': 'text'}, inplace=True)

    df_falcon['generated'] = np.ones(df_falcon.shape[0])
    df_llama['generated'] = np.ones(df_llama.shape[0])

    df_extras = pd.concat([df_palm[['text', 'generated']], df_falcon[['text', 'generated']], df_llama[['text', 'generated']]], axis=0)
    return df_extras
