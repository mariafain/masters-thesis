import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



def get_tokenized_sequences(x_train, x_valid, x_test):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(pd.concat([x_train, x_valid], axis=0))

    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)
    sequences_valid = tokenizer.texts_to_sequences(x_valid)

    return (sequences_train, sequences_valid, sequences_test), tokenizer

def pad_all_sequences(sequences_train, sequences_valid, sequences_test, max_seq_len, truncating='post'):
    x_train = pad_sequences(sequences_train, maxlen=max_seq_len, truncating='post')
    x_test = pad_sequences(sequences_test, maxlen=max_seq_len, truncating='post')
    x_valid = pad_sequences(sequences_valid, maxlen=max_seq_len, truncating='post')

    return x_train, x_valid, x_test

def get_embeddings(tokenizer, vocab_size, path_to_glove, embedding_dim):
    embeddings_index = {}
    # read word vectors
    with open(path_to_glove, encoding='utf8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    # assign word vectors to our dictionary/vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            # this includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

def prepare_for_modeling(x_train, x_valid, x_test, max_seq_len, path_to_glove, embedding_dim):
    (sequences_train, sequences_valid, sequences_test), tokenizer = get_tokenized_sequences(x_train, x_valid, x_test)

    x_train, x_valid, x_test = pad_all_sequences(sequences_train, sequences_valid, sequences_test, max_seq_len)
    vocab_size = len(tokenizer.index_word) + 1

    embedding_matrix = get_embeddings(tokenizer, vocab_size, path_to_glove, embedding_dim)

