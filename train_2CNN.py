from preprocess import define_preprocess_func, data_pipeline, test_data_generator
from plot import plot_training
from metrics import evaluation_on_generator, evaluation
from training import TimeLine, train
from CNNs import init_2CNN_model, init_shared_model

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from time import time
import datetime
import cv2
import glob
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import xception
from keras.applications import imagenet_utils
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers, models
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ModelCheckpoint, CSVLogger

from keras import backend as K
import numpy as np
np.random.seed(2018)
K.clear_session()
K.set_image_dim_ordering('tf')


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
            self.model_name = 'shared' #customized/vgg16/19
            self.version_as_suffix = 'try'
            self.batch_size = 64
            self.epochs = 100
            self.show_plot = False

    args = init_args()

    # build data
    preprocess_input = define_preprocess_func(args)

    a_X_train, a_y_train = data_pipeline(args.a_train_dir, preprocess_input)
    a_y_train = to_categorical(a_y_train, args.num_class)

    b_X_train, b_y_train = data_pipeline(args.b_train_dir, preprocess_input)
    b_y_train = to_categorical(b_y_train, args.num_class)


    # init CNN
    model = init_shared_model(args)
    #model = init_2CNN_model(args)
    model.summary()

    # training
    X_train = [a_X_train, b_X_train]
    y_train = [a_y_train, b_y_train]
    trained_model, tensorboard_path, checkpoint_path, csv_path, time_path = train(model, X_train, y_train, args)
    if args.show_plot:
        plot_training(args, csv_path)
        # raise error because cols name has changed

    # evaluate on test data
    a_X_test, a_y_test = data_pipeline(args.a_test_dir, preprocess_input)
    a_y_test = to_categorical(a_y_test, args.num_class)

    b_X_test, b_y_test = data_pipeline(args.b_test_dir, preprocess_input)
    b_y_test = to_categorical(b_y_test, args.num_class)

    print('--------------------')
    print('predicting test dateset')
    saved_model = models.load_model(checkpoint_path)
    X_test = [a_X_test, b_X_test]
    y_test = [a_y_test, b_y_test]
    score = model.evaluate(X_test, y_test)
    print(list(zip(model.metrics_names, score)))

# /Users/San/Projects/ImageClassifier/callbacks/2CNN_10_28_50_64_test.h5
# loss 1: 10
# ('a_pred_acc', 0.521), ('b_pred_acc', 0.964)
