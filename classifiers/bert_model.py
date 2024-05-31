import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam


class MyBertClassifier:
    def __init__(self, name, params_dict, tfhub_handle_encoder, tfhub_handle_preprocess) -> None:
        self.name = name
        self.params_dict = params_dict.copy()
        self.tfhub_handle_encoder = tfhub_handle_encoder
        self.tfhub_handle_preprocess = tfhub_handle_preprocess
        self.model = None
        self.history = None

    def build_classifier(self):
        text_input = Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = Dropout(0.1)(net)
        net = Dense(1, activation=None, name='classifier')(net)

        self.model = tf.keras.Model(text_input, net)
    
    def compile_classifier(self):
        if self.model:
            loss = BinaryCrossentropy(from_logits=True)
            metrics = BinaryAccuracy('accuracy')
            optimizer = Adam(learning_rate=self.params_dict['learning_rate'])

            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=[metrics])
        else:
            print('You cannot compile the model before you have built it. Please call the `build_classifier` function first.')

    def fit_model(self, x_train, y_train, x_valid, y_valid, save_model=True):
        self.history = self.model.fit(x=x_train,
                                      y=y_train,
                                      validation_data=(x_valid, y_valid),
                                      epochs=self.params_dict['epochs'])
        if save_model:
            self.model.save(os.path.join(os.getcwd(), 'models', self.name + '.h5'))
