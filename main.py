import os
import numpy as np
import pandas as pd

from bilstm import Bilstm, get_tokenized_sequences, pad_all_sequences, get_embeddings
from preprocessing import get_small_df, preprocess_df
from utils import stratified_split, get_x_y, MAX_SEQ_LEN
from validation import validate_model


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'transformed_wiki_data.csv'), header=0)
    
    # preprocessing
    df = get_small_df(df)
    df = preprocess_df(df)

    # split data
    train_set, valid_set, test_set = stratified_split(df)
    x_train, y_train = get_x_y(train_set)
    x_valid, y_valid = get_x_y(valid_set)
    x_test, y_test = get_x_y(test_set)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
    y_valid = np.asarray(y_valid).astype('float32').reshape((-1,1))

    # prepare data for modeling with bilstm
    (sequences_train, sequences_valid, sequences_test), tokenizer = get_tokenized_sequences(x_train, x_valid, x_test)
    x_train, x_valid, x_test = pad_all_sequences(sequences_train, sequences_valid, sequences_test, MAX_SEQ_LEN)
    vocab_size = len(tokenizer.index_word) + 1

    path_to_glove = os.path.join(os.getcwd(), 'downloads', 'glove.6B.200d.txt')
    embedding_dim = 200
    embedding_matrix = get_embeddings(tokenizer, vocab_size, path_to_glove, embedding_dim)

    # modeling
    model_name = 'bilstm_small_v1'
    params_dict = {'dropout': 0.3,
                   'rec_dropout': 0.3,
                   'learning_rate': 0.006,
                   'patience': 4,
                   'units': 128,
                   'batch': 256,
                   'epochs': 15}
    bilstm = Bilstm(params_dict)
    bilstm.build_model(vocab_size=vocab_size,
                       input_len=x_train.shape[1],
                       embedding_dim=embedding_dim,
                       embedding_matrix=embedding_matrix)
    bilstm.fit_model(x_train, y_train, x_valid, y_valid)
    bilstm.model.save(os.path.join(os.getcwd(), 'models', model_name))

    # validation
    validate_model(bilstm, x_test, y_test)
