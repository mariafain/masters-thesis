import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score


def get_predictions(classifier, x_test):
    pred_probas = classifier.model.predict(x_test)
    predictions = np.where(pred_probas.max(axis=-1) > 0.5, 1, 0)
    return pred_probas, predictions

def acc_loss(history, save: bool):
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

    if save:
        plt.savefig('acc_loss.png')
    plt.show()

def conf_matrix(y_true, y_pred, save):
    cm = confusion_matrix(np.array(y_true), y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    if save:
        plt.savefig('conf_matrix.png')
    plt.show()

def roc_score(model_name, y_true, y_pred_proba, save):
    """
    Plots the ROC curve for the given model and data. Calculates and prints the area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f'AUC: {auc}')
    plt.plot(fpr, tpr, label=model_name+" AUC="+str(round(auc, 3)))
    plt.legend(loc='best')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve for {model_name}')
    if save:
        plt.savefig('roc.png')
    plt.show()

def validate_classifier(classifier, x_test, y_true, new_data=False, save=False):
    print('#### VALIDATION ####')
    predictions_proba, y_pred = get_predictions(classifier, x_test)
    
    if not new_data:
        acc_loss(classifier.history, save)
        roc_score(classifier.name, y_true, predictions_proba.flatten(), save)
    print('Classification report:')
    print(classification_report(y_true, y_pred, zero_division=0))
    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('Recall: ', recall_score(y_true, y_pred))
    print('F1-score: ', f1_score(y_true, y_pred))

    conf_matrix(y_true, y_pred, save)

# def validate_model_bilstm(bilstm, x_test, y_true):
#     print('#### VALIDATION ####')
#     print('Test set evaluation:')
#     bilstm.model.evaluate(x_test, y_true, verbose=1)

#     y_pred = get_predictions_bilstm(bilstm, x_test)
#     print('Classification report:')
#     print(classification_report(y_true, y_pred))

#     acc_loss(bilstm.history)
#     conf_matrix(y_true, y_pred.flatten())
   