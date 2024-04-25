import io
import json
import logging
import os
import time

import numpy as np
import pandas as pd

import bilstm
from preprocessing import get_small_df, preprocess_df
from utils import stratified_split, get_x_y, MAX_SEQ_LEN, init_logger
from validation import validate_model_bilstm


if __name__ == "__main__":
    init_logger(__file__)
    logger = logging.getLogger(__file__)

    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'transformed_wiki_data.csv'), header=0)
    
    # preprocessing
    start_time = time.time()
    logger.info(f'Starting preprocessing...')
    df = get_small_df(df)
    df = preprocess_df(df)
    logger.info(f'Done with preprocessing. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    # split data
    train_set, valid_set, test_set = stratified_split(df)
    x_train, y_train = get_x_y(train_set)
    x_valid, y_valid = get_x_y(valid_set)
    x_test, y_test = get_x_y(test_set)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
    y_valid = np.asarray(y_valid).astype('float32').reshape((-1,1))

    # prepare data for modeling with bilstm
    (sequences_train, sequences_valid, sequences_test), tokenizer = bilstm.get_tokenized_sequences(x_train, x_valid, x_test)
    x_train, x_valid, x_test = bilstm.pad_all_sequences(sequences_train, sequences_valid, sequences_test, MAX_SEQ_LEN)
    vocab_size = len(tokenizer.index_word) + 1

    path_to_glove = os.path.join(os.getcwd(), 'downloads', 'glove.6B.200d.txt')
    embedding_dim = 200
    embedding_matrix = bilstm.get_embeddings(tokenizer, vocab_size, path_to_glove, embedding_dim)

    # modeling
    model_name = 'bilstm_small_v1.keras'
    params_dict = {'dropout': 0.3,
                   'rec_dropout': 0.3,
                   'learning_rate': 0.006,
                   'patience': 4,
                   'units': 128,
                   'batch': 256,
                   'epochs': 15}
    bilstm_model = bilstm.Bilstm(model_name, params_dict)
    bilstm_model.build_model(vocab_size=vocab_size,
                       input_len=x_train.shape[1],
                       embedding_dim=embedding_dim,
                       embedding_matrix=embedding_matrix)
    
    start_time = time.time()
    logger.info(f'Training the model...')
    bilstm_model.fit_model(x_train, y_train, x_valid, y_valid)
    logger.info(f'Training is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')

    bilstm_model.model.save(os.path.join(os.getcwd(), 'models', bilstm_model.name))
    tokenizer_json = tokenizer.to_json()
    with io.open('models/tokenizer_lstm.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # validation
    start_time = time.time()
    logger.info(f'Validating the model...')
    validate_model_bilstm(bilstm_model, x_test, y_test)
    logger.info(f'Validation is done. Time elapsed: {((time.time() - start_time)/60):.1f}min.')
