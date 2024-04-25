import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

MAX_SEQ_LEN = 164


def init_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

def get_x_y(df: pd.DataFrame, feature: str='preprocessed_text', target_class: str='generated'):
    x = df[feature]
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


# probably delete this
def split_data(df: pd.DataFrame, features: List[str], target_class: str='generated', train_size: float=0.6, test_size: float=0.2):
    """
    This function ...
    """
    train_size = int(math.floor(df.shape[0] * train_size))
    test_size = int(math.floor(df.shape[0] * test_size))

    train_set = df[:train_size]
    valid_set = df[train_size:train_size + test_size]
    test_set = df[train_size + test_size:]

    x_train, y_train = get_x_y(train_set, features, target_class)
    x_valid, y_valid = get_x_y(valid_set, features, target_class)
    x_test, y_test = get_x_y(test_set, features, target_class)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
