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

    sequences_train = tokenizer.texts_to_sequences(x_train.values.tolist())
    sequences_test = tokenizer.texts_to_sequences(x_test.values.tolist())
    sequences_valid = tokenizer.texts_to_sequences(x_valid.values.tolist())

    return (sequences_train, sequences_valid, sequences_test), tokenizer

def pad_all_sequences(sequences_train, sequences_valid, sequences_test, max_seq_len, truncating='post'):
    x_train = pad_sequences(sequences_train, maxlen=max_seq_len, truncating=truncating)
    x_test = pad_sequences(sequences_test, maxlen=max_seq_len, truncating=truncating)
    x_valid = pad_sequences(sequences_valid, maxlen=max_seq_len, truncating=truncating)

    return x_train, x_valid, x_test

def get_embeddings(tokenizer, vocab_size, path_to_glove, embedding_dim):
    hits = 0
    misses = 0
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


class Bilstm:
    def __init__(self, name, params_dict) -> None:
        self.name = name
        self.params_dict = params_dict.copy()
        self.model = None
        self.history = None

    def build_model(self, vocab_size: int, input_len: int, embedding_dim: int, embedding_matrix: np.ndarray) -> None:
        model = Sequential()
        model.add(Embedding(vocab_size,
                            embedding_dim,
                            input_length=input_len,
                            weights=[embedding_matrix],
                            trainable=False))
        model.add(Bidirectional(LSTM(units=self.params_dict['units'],
                                    recurrent_dropout=self.params_dict['rec_dropout'],)))
                                    # return_sequences=True)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer=Adam(learning_rate=self.params_dict['learning_rate']),
                    metrics=['accuracy'])
        model.summary()
        self.model = model
    
    def fit_model(self, x_train, y_train, x_valid, y_valid) -> None:
        if not self.model:
            print('The model needs to be built before it is trained!')
            return
        
        callback = EarlyStopping(
            monitor="val_loss",
            patience=self.params_dict['patience'],
            restore_best_weights=True)

        self.history = self.model.fit(x_train,
                                    y_train,
                                    validation_data=(x_valid, y_valid),
                                    verbose=1,
                                    batch_size=self.params_dict['batch'],
                                    epochs=self.params_dict['epochs'],
                                    callbacks=[callback])
