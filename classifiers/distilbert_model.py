import os

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification


class MyDistilbertClassifier:
    def __init__(self, name, params_dict, max_seq_len, num_labels) -> None:
        self.name = name
        self.params_dict = params_dict.copy()
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = None
        self.history = None

    def preprocess_data(self, x_data):
        return self.tokenizer(x_data, max_length=self.max_seq_len, padding=True, truncation=True, return_tensors='tf')

    def build_classifier(self):
        distilbert = TFDistilBertForSequenceClassification.from_pretrained(self.name, num_labels=self.num_labels)

        input_ids = Input(shape=(self.max_seq_len,), name='input_ids', dtype='int32')
        mask = Input(shape=(self.max_seq_len,), name='attention_mask', dtype='int32')

        embeddings = distilbert(input_ids, attention_mask=mask)[0]
        y = Dense(1, activation='sigmoid', name='outputs')(embeddings)

        self.model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    
    def compile_classifier(self):
        if self.model:
            loss = BinaryCrossentropy()
            metrics = BinaryAccuracy('accuracy')
            optimizer = Adam(learning_rate=self.params_dict['learning_rate'])

            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=[metrics])
        else:
            print('You cannot compile the model before you have built it. Please call the `build_classifier` function first.')

    def fit_model(self, x_train, y_train, x_valid, y_valid, save_model=True):
        self.history = self.model.fit((x_train['input_ids'], x_train['attention_mask']),
                                      y_train,
                                      validation_data=((x_valid['input_ids'], x_valid['attention_mask']), y_valid),
                                      epochs=self.params_dict['epochs'],
                                      batch_size=self.params_dict['batch'])
        if save_model:
            self.model.save(os.path.join(os.getcwd(), 'models', self.name + '.h5'))
