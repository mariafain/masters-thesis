import logging
import os
import sys
import time

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from tensorflow.keras.models import load_model

from classifiers.roberta_model import MyRobertaClassifier
from preprocessing import get_small_df, preprocess_df, prepare_data_for_predicting
from utils import stratified_split, get_x_y, init_logger, load_extra_data, MAX_SEQ_LEN
from validation import validate_classifier


if __name__ == "__main__":
    init_logger(__file__)
    logger = logging.getLogger(__file__)

    # define hyperparameters
    model_name = 'roberta-base'
    params_dict = {'learning_rate': 1e-5,
                   'epochs': 3,
                   'batch': 16}

    # load data
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'transformed_wiki_data.csv'), header=0)
    
    start_time = time.time()
    logger.info(f'Starting preprocessing...')
    # preprocessing
    df = get_small_df(df)
    df = preprocess_df(df)
    logger.info(f'Done with preprocessing. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # # if you have already processed the data, skip the load data and preprocessing data and uncomment this code
    # df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'preprocessed_df.csv'), header=0)

    # split data
    train_set, valid_set, test_set = stratified_split(df)
    # shuffle rows
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    valid_set = valid_set.sample(frac=1).reset_index(drop=True)

    x_train, y_train = get_x_y(train_set)
    x_valid, y_valid = get_x_y(valid_set)
    x_test, y_test = get_x_y(test_set)

    x_train = x_train.values.tolist()
    x_test = x_test.values.tolist()
    x_valid = x_valid.values.tolist()
    
    # modeling
    roberta_classifier = MyRobertaClassifier(model_name, params_dict, MAX_SEQ_LEN, num_labels=2)
    roberta_classifier.build_classifier()
    roberta_classifier.compile_classifier()
    roberta_classifier.model.summary()

    # preprocess data for roberta
    tokenized_train = roberta_classifier.preprocess_data(x_train)
    tokenized_valid = roberta_classifier.preprocess_data(x_valid)
    tokenized_test = roberta_classifier.preprocess_data(x_test)

    start_time = time.time()
    logger.info(f'Training the model...')
    roberta_classifier.fit_model(tokenized_train, y_train, tokenized_valid, y_valid)
    logger.info(f'Training is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # # if you want to reload the model you can do it by uncommenting the following code
    # reloaded_roberta_classifier = load_model((os.path.join(os.getcwd(), 'models', model_name + '.h5')), custom_objects={'TFRobertaForSequenceClassification': TFRobertaForSequenceClassification})
    
    # validation
    start_time = time.time()
    logger.info(f'Validating the model...')
    validate_classifier(roberta_classifier, dict(tokenized_test), y_test)
    logger.info(f'Validation is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # test on new data - essays and extra LLMs
    df_essays = pd.read_csv(os.path.join(os.getcwd(), 'data', 'extras', 'train_essays.csv'), header=0)
    x_essays, y_essays = prepare_data_for_predicting(df_essays)
    # # if the model is reloaded you need to load the tokenizer as well
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # x_essays = tokenizer(x_essays)
    x_essays = roberta_classifier.preprocess_data(x_essays.values.tolist())
    validate_classifier(roberta_classifier, dict(x_essays), y_essays, new_data=True)

    df_extras = load_extra_data(os.path.join('data', 'extras'))
    x_extras, y_extras = prepare_data_for_predicting(df_extras)
    # x_extras = tokenizer(x_extras)
    x_extras = roberta_classifier.preprocess_data(x_extras.values.tolist())
    validate_classifier(roberta_classifier, dict(x_extras), y_extras, new_data=True)
