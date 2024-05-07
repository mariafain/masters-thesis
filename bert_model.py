import ktrain


def preprocess_bert_data(x_train, y_train, x_valid, y_valid, class_names, max_seq_len, max_features):
    (x_train, y_train), (x_valid, y_valid), preproc = ktrain.text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                    x_test=x_valid, y_test=y_valid,
                                                                    preprocess_mode='bert',
                                                                    class_names=class_names,
                                                                    maxlen=max_seq_len,
                                                                    max_features=max_features)
    return x_train, y_train, x_valid, y_valid, preproc


class Bert:
    def __init__(self, name, params_dict, preproc) -> None:
        self.name = name
        self.params_dict = params_dict.copy()
        self.preproc = preproc
        self.learner = None
        self.predictor = None

    def build_learner(self, x_train, y_train, x_valid, y_valid):
        model = ktrain.text.text_classifier('bert',
                                            train_data=(x_train, y_train),
                                            preproc=self.preproc)
        
        learner = ktrain.get_learner(model, 
                                     train_data=(x_train, y_train),
                                     val_data=(x_valid, y_valid),
                                     batch_size=self.params_dict['batch'])
        self.learner = learner

    def fit_model(self):
        self.learner.fit_onecycle(self.params_dict['learning_rate'],
                                  self.params_dict['epochs'])
        self.predictor = ktrain.get_predictor(self.learner.model, self.preproc)
    