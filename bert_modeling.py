import logging
import os
import time

import pandas as pd

import bert
from preprocessing import get_small_df, preprocess_df
from utils import stratified_split, get_x_y, MAX_SEQ_LEN, init_logger
from validation import validate_model_bert


if __name__ == "__main__":
    init_logger(__file__)
    logger = logging.getLogger(__file__)

    model_name = 'bert_model_latest.keras'
    params_dict = {'learning_rate': 2e-5,
                   'epochs': 3,
                   'batch': 16,
                   'max_features': 35000}
    class_names = ['not generated', 'generated']

    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'transformed_wiki_data.csv'), header=0)
    
    start_time = time.time()
    logger.info(f'Starting preprocessing...')
    # preprocessing
    df = get_small_df(df)
    df = preprocess_df(df)
    logger.info(f'Done with preprocessing. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # split data
    train_set, valid_set, test_set = stratified_split(df)
    x_train, y_train = get_x_y(train_set)
    x_valid, y_valid = get_x_y(valid_set)
    x_test, y_test = get_x_y(test_set)

    x_train = x_train.tolist()
    x_test = x_test.tolist()
    x_valid = x_valid.tolist()

    x_train, y_train, x_valid, y_valid, preproc = bert.preprocess_bert_data(x_train,
                                                                            y_train,
                                                                            x_valid,
                                                                            y_valid,
                                                                            class_names,
                                                                            MAX_SEQ_LEN,
                                                                            params_dict['max_features'])
    
    # modeling
    bert_model = bert.Bert(model_name, params_dict, preproc)
    bert_model.build_learner(x_train, y_train, x_valid, y_valid)

    start_time = time.time()
    logger.info(f'Training the model...')
    bert_model.fit_model()
    logger.info(f'Training is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # save the model
    bert_model.predictor.save(os.path.join(os.getcwd(), 'models', bert_model.name))
    
    #validation
    start_time = time.time()
    logger.info(f'Validating the model...')
    validate_model_bert(bert_model, x_test, y_test)
    logger.info(f'Validation is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')
