import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def get_predictions_bilstm(bilstm, x_test):
    predictions = bilstm.model.predict(x_test)
    predictions = np.where(predictions.max(axis=-1) > 0.5, 1, 0)
    return predictions

def get_predictions_bert(bert, x_test):
    """
    The argument `bert` is an instance of the class Bert. The function returns the binary predictions
    the bert model makes for the array of preprocessed texts `x_test`.
    """
    predictions = bert.predictor.predict(x_test)
    preds_bin = [1 if el == 'generated' else 0 for el in predictions]
    return preds_bin

def acc_loss(history):
    # Visualize Loss & Accuracy

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def conf_matrix(y_true, y_pred):
    cm = confusion_matrix(np.array(y_true), y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def validate_model_bilstm(bilstm, x_test, y_true):
    print('#### VALIDATION ####')
    print('Test set evaluation:')
    bilstm.model.evaluate(x_test, y_true, verbose=1)

    y_pred = get_predictions_bilstm(bilstm, x_test)
    print('Classification report:')
    print(classification_report(y_true, y_pred))

    acc_loss(bilstm.history)
    conf_matrix(y_true, y_pred.flatten())

def validate_model_bert(bert, x_test, y_true):
    print('#### VALIDATION ####')

    y_pred = get_predictions_bert(bert, x_test)
    print('Classification report:')
    print(classification_report(y_true, y_pred))

    conf_matrix(y_true, y_pred)
    