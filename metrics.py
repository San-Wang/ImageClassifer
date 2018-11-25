import os
from keras import models
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from preprocess import *


########################################
#  evaluate model acc
########################################


def evaluation(model, X_test, y_test):
    """ evaluate
    args:
        :param model: loded model
        :param X_test: ndarray [batch, size, size, channel]
        :param y_test: adarray [batch,]
    :return:
    """
    y_logit = model.predict_on_batch(X_test)
    y_pred = np.argmax(y_logit, 1)
    print('accuracy:{}\n'.format(accuracy_score(y_test, y_pred)))
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('classification report:')
    print(classification_report(y_test, y_pred))

def evaluation_on_generator(data_dir, model, args):
    test_generator = test_data_generator(data_dir, args)
    # prediction = model.predict_generator(test_generator) # shape:(#sample, 4)
    # argmax
    score = model.evaluate_generator(test_generator)
    metrics = list(zip(model.metrics_names, score))  # [('loss',int),('acc', int)]
    print(metrics)

    return metrics



if __name__ == '__main__':
    class init_args(object):
        def __init__(self):
            self.callbacks_dir = '/Users/San/Projects/ImageClassifier/callbacks'
            self.a_train_dir = '/Users/San/Projects/ImageClassifier/data/CIFAR10/Train'
            self.a_val_dir = None
            self.a_test_dir = '/Users/San/Projects/ImageClassifier/data/CIFAR10/Test'
            self.b_train_dir = '/Users/San/Projects/ImageClassifier/data/MNIST/Train'
            self.b_val_dir = None
            self.b_test_dir = '/Users/San/Projects/ImageClassifier/data/MNIST/Test'
            self.num_class = 10
            self.img_size = 28  # resize target
            self.channels = 1
            self.pretrain = False # choose to use pretrained model or training from scratch
            self.model_name = '2CNN' #customized/vgg16/19
            self.version_as_suffix = 'test'
            self.batch_size = 64
            self.epochs = 50
            self.show_plot = False



    args = init_args()
    model_path = '/Users/San/Projects/ImageClassifier/callbacks/2CNN_10_28_1_64_try3.h5'
    model = models.load_model(model_path)



    preprocess_input = define_preprocess_func(args)
    a_X_test, a_y_test = data_pipeline(args.a_test_dir, preprocess_input)
    a_y_test = LabelBinarizer().fit_transform(a_y_test)
    b_X_test, b_y_test = data_pipeline(args.b_test_dir, preprocess_input)
    b_y_test = LabelBinarizer().fit_transform(b_y_test)
    X_test = [a_X_test, b_X_test]
    y_test = [a_y_test, b_y_test]


    score = model.evaluate(X_test, y_test)
    print(list(zip(model.metrics_names, score)))
