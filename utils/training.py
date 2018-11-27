import datetime
from time import time
import os
import pandas as pd
import numpy as np

from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, CSVLogger


class TimeLine(Callback):
    """ add current datetime at the beginning of epoch into logs dict"""
    def on_train_begin(self, logs):
        self.start_datetime = []
        self.start_unix = []
        self.end_datetime = []
        self.end_unix = []
    def on_epoch_begin(self, epoch, logs={}):
        self.start_datetime.append(datetime.datetime.now())
        self.start_unix.append(time())

    def on_epoch_end(self, epoch, logs={}):
        self.end_datetime.append(datetime.datetime.now())
        self.end_unix.append(time())

def train_on_generator(model, train_generator, validation_generator, args):
    """
    args:
        :param model: keras.models.Model().compile()
        :param train_generator:
        :param validation_generator:
        :param:
            args.callbacks_dir
            args.model_name
            args.num_class
            args.img_size
            args.epochs
            args.batch_size
            args.version_as_suffix

    return:
        :model: trained model
        :tensorboard_path:
        :checkpoint_path:
        :csv_path:
        :timeline_path:
    """
    # colab root dir, need manually create for ModelCheckpoint
    root_dir = args.callbacks_dir
    # create root_dir if not exists, where will save all log files in
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    #############################
    # <keras.callbacks.TensorBoard>
    #   function: save tensorboard event under root_dir/pretrained_{args.model_name}_{args.num_class}_{args.img_size}_{args.epochs}_{args.batch_size}_{args.version_as_suffix}/
    # Arguments:
    #   :histogram_freq:  frequency (in epochs) at which to compute activation and weight histograms for the layers of the model
    #   :write_graph: whether to visualize the graph in TensorBoard
    #   :write_grads: whether to visualize gradient histograms in TensorBoard. (histogram_freq must be greater than 0)
    #   :write_images: whether to write model weights to visualize as image in TensorBoard
    #   :
    # <keras.callbacks.ModelCheckpoint>
    #   function: save model as root_dir/pretrained_{args.model_name}_{args.num_class}_{args.img_size}_{args.epochs}_{args.batch_size}_{args.version_as_suffix}.h5
    #############################

    model_name = '{}_{}_{}_{}_{}_{}'.format(args.model_name,
                                                                args.num_class,
                                                                args.img_size,
                                                                args.epochs,
                                                                args.batch_size,
                                                                args.version_as_suffix)

    # path -> folder
    tensorboard_path = '{}/{}'.format(root_dir, model_name)
    # path -> model file
    checkpoint_path = '{}/{}.h5'.format(root_dir, model_name)
    # path -> csv file
    csv_path = '{}/{}.log'.format(root_dir, model_name)

    timeline_path = '{}/{}_time.csv'.format(root_dir, model_name)

    tensorboard = TensorBoard(log_dir=tensorboard_path,
                              histogram_freq=0, write_graph=True, write_grads=False, write_images=True)


    # ModelCheckpoint can't create parent folder automatically
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger(csv_path)

    timeline = TimeLine()

    callbacks_list = [checkpoint, tensorboard, csv_logger, timeline]

    stepsPerEpoch = train_generator.samples // args.batch_size
    if validation_generator is None:
        validationSteps = None
    else:
        validationSteps = validation_generator.samples // args.batch_size

    start = time()

    model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,  # It should typically be equal to #samples of dataset / batch size
        epochs=args.epochs,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=validationSteps,
        verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch
    )

    total_time = (time() - start) / 60
    print('model {} finished in {:.2f} mins'.format(checkpoint_path, total_time))

    timeline_df = pd.DataFrame({
        'epoch': range(args.epochs),
        'start_datetime': timeline.start_datetime,
        'end_datetime': timeline.end_datetime,
        'start_unix': timeline.start_unix,
        'end_unix': timeline.end_unix
    })
    timeline_df.to_csv(timeline_path, index=False)

    return model, tensorboard_path, checkpoint_path, csv_path, timeline_path

def train_2CNN(model, a_X_train, a_y_train, b_X_train, b_y_train, args):

    root_dir = args.callbacks_dir
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    model_name = '{}_{}_{}_{}_{}_{}'.format(args.model_name,
                                            args.num_class,
                                            args.img_size,
                                            args.epochs,
                                            args.batch_size,
                                            args.version_as_suffix)

    tensorboard_path = '{}/{}'.format(root_dir, model_name)
    checkpoint_path = '{}/{}.h5'.format(root_dir, model_name)
    csv_path = '{}/{}.log'.format(root_dir, model_name)
    timeline_path = '{}/{}_time.csv'.format(root_dir, model_name)

    tensorboard = TensorBoard(log_dir=tensorboard_path,
                              histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger(csv_path)
    timeline = TimeLine()

    callbacks_list = [checkpoint, tensorboard, csv_logger, timeline]

    start = time()

    if args.a_val_dir and args.b_val_dir:
        pass

    else:
        model.fit(
            [a_X_train, b_X_train],
            [a_y_train, b_y_train],
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks_list,
            verbose=2
        )

    total_time = (time() - start) / 60
    print('model {} finished in {:.2f} mins'.format(checkpoint_path, total_time))

    return model, tensorboard_path, checkpoint_path, csv_path, timeline_path

def train(model, X, y, args):
    """ train CNN model
    for multi-input/output, use X = [x_train_1, x_train_2]
                                y = [y_train_1, y_train_2]
    args:
        :param model: keras.models.Model().compile()
        :param train_generator:
        :param validation_generator:
        :param:
            args.callbacks_dir
            args.model_name
            args.num_class
            args.img_size
            args.epochs
            args.batch_size
            args.version_as_suffix

    return:
        :model: trained model
        :tensorboard_path:
        :checkpoint_path:
        :csv_path:
        :timeline_path:
    """

    root_dir = args.callbacks_dir
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    model_name = '{}_{}_{}_{}_{}_{}'.format(args.model_name,
                                                                args.num_class,
                                                                args.img_size,
                                                                args.epochs,
                                                                args.batch_size,
                                                                args.version_as_suffix)

    # path -> folder
    tensorboard_path = '{}/{}'.format(root_dir, model_name)
    # path -> model file
    checkpoint_path = '{}/{}.h5'.format(root_dir, model_name)
    # path -> csv file
    csv_path = '{}/{}.log'.format(root_dir, model_name)

    timeline_path = '{}/{}_time.csv'.format(root_dir, model_name)

    tensorboard = TensorBoard(log_dir=tensorboard_path,
                              histogram_freq=0, write_graph=True, write_grads=False, write_images=True)


    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger(csv_path)

    timeline = TimeLine()

    callbacks_list = [checkpoint, tensorboard, csv_logger, timeline]

    start = time()

    model.fit(
        X,
        y,
        shuffle = True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks_list,
        verbose=2
    )

    total_time = (time() - start) / 60
    print('model {} finished in {:.2f} mins'.format(checkpoint_path, total_time))

    timeline_df = pd.DataFrame({
        'epoch': range(args.epochs),
        'start_datetime': timeline.start_datetime,
        'end_datetime': timeline.end_datetime,
        'start_unix': timeline.start_unix,
        'end_unix': timeline.end_unix
    })
    timeline_df.to_csv(timeline_path, index=False)

    return model, tensorboard_path, checkpoint_path, csv_path, timeline_path
