import logging
import os
import sys
import time

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

from classifiers.bert_model import MyBertClassifier
from preprocessing import get_small_df, preprocess_df, prepare_data_for_predicting
from utils import stratified_split, get_x_y, init_logger, load_extra_data
from validation import validate_classifier


if __name__ == "__main__":
    init_logger(__file__)
    logger = logging.getLogger(__file__)

    # define hyperparameters
    model_name = 'bert_classifier'
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    params_dict = {'learning_rate': 2e-5,
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
    
    # modeling
    bert_classifier = MyBertClassifier(model_name, params_dict, tfhub_handle_encoder, tfhub_handle_preprocess)
    bert_classifier.build_classifier()
    bert_classifier.compile_classifier()
    bert_classifier.model.summary()

    start_time = time.time()
    logger.info(f'Training the model...')
    bert_classifier.fit_model(x_train, y_train, x_valid, y_valid)
    logger.info(f'Training is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # # if you want to reload the model you can do it by uncommenting the following code
    # reloaded_bert_classifier = load_model((os.path.join(os.getcwd(), 'models', model_name + '.h5')), custom_objects={'KerasLayer': hub.KerasLayer})
    
    # validation
    start_time = time.time()
    logger.info(f'Validating the model...')
    validate_classifier(bert_classifier, x_test, y_test)
    logger.info(f'Validation is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # test on new data - essays and extra LLMs
    df_essays = pd.read_csv(os.path.join(os.getcwd(), 'data', 'extras', 'train_essays.csv'), header=0)
    x_essays, y_essays = prepare_data_for_predicting(df_essays)
    validate_classifier(bert_classifier, x_essays, y_essays, new_data=True)

    df_extras = load_extra_data(os.path.join('data', 'extras'))
    x_extras, y_extras = prepare_data_for_predicting(df_extras)
    validate_classifier(bert_classifier, x_extras, y_extras, new_data=True)
