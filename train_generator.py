import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from time import time
import datetime
import cv2
import glob
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import xception
from keras.applications import imagenet_utils
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import optimizers, models
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ModelCheckpoint, CSVLogger

from keras import backend as K
import numpy as np
np.random.seed(2018)
K.clear_session()
K.set_image_dim_ordering('tf')



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--callbacks_dir", type=str, required=True, default='./callbacks', help="directory path to save callbacks files")
    ap.add_argument("--train_dir", type=str, required=True, help="(required) the train data directory")
    ap.add_argument("--val_dir", type=str, default=None, help="the validation data directory")
    ap.add_argument("test_dir", type=str, default=None, help="test data directory")
    ap.add_argument("--num_class", type=int, required=True,help="number of classes to classify")
    ap.add_argument("--img_size", type=int, default=224, help="target image width/height size")
    ap.add_argument("-channels", type=int, required=True, help="image channels: 1 for grey, 3 for color")
    ap.add_argument("--pretrain", type=bool, default=False, help="whether to use pretrain model architecture")
    ap.add_argument("--model_name", type=str, default='vgg16', help="model name")
    ap.add_argument("--version_as_suffix", type=str, default='test', help="version_as_suffix for model name")
    ap.add_argument("--batch_size", type=int, default=16, help="training batch size")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs")
    ap.add_argument("--show_plot", type=bool, default=True, help="whether plot acc&loss against epoch")

    args = ap.parse_args()
    return args


def define_preprocess_input(args):
    """ define preprocess_input from args param
    args:
        :args.pretrain: boolean
        :args.model_name: str
    :return:
        :preprocess_input: function takes image and return image
    """
    MODELS = {
        "vgg16": vgg16.VGG16,
        "vgg19": vgg19.VGG19,
        "inception": inception_v3.InceptionV3,
        "xception": xception.Xception,
        "resnet50": resnet50.ResNet50
    }

    # when use customized structure
    # if not args.pretrain:
    #     def preprocess_input(x):
    #         img = imagenet_utils.preprocess_input(image.img_to_array(x))  # scale pixels between -1 and 1, sample-wise: x /= 127.5, x -= 1
    #         return image.array_to_img(img)
    if not args.pretrain:
        # when args.channels = 3
        if args.channels == 3:
            def preprocess_input(x):
                img = imagenet_utils.preprocess_input(image.img_to_array(x)) #scale pixels between -1 and 1, sample-wise: x /= 127.5, x -= 1
                return image.array_to_img(img)
        # when channels = 1
        elif args.channels == 1:
            def preprocess_input(x):
                img = image.img_to_array(x)
                # resize
                img = cv2.resize(img, (args.img_size, args.img_size), interpolation = cv2.INTER_CUBIC)
                img = image.img_to_array(img) # img_to_array able to make ndarray [28,28] -> [28,28,1]
                # normalization
                img /= 225.0
                img = image.array_to_img(img) #input img rank have to be 3
                return img

    elif args.model_name in ('vgg16', 'vgg19', 'resnet50'):
        def preprocess_input(x):
            img = imagenet_utils.preprocess_input(image.img_to_array(x))
            return image.array_to_img(img)

    elif args.model_name in ("inception", "xception"):
        def preprocess_input(x):
            img = inception_v3.preprocess_input(image.img_to_array(x))
            return image.array_to_img(img)

    elif args.pretrain and args.model_name not in MODELS:
        print('input pretrain model preprocessing has not been pre-defined yet')
        raise AttributeError

    return preprocess_input


def test_data_generator(root_dir, args):
    """ read test image data from directory using keras API
    including define preprocess_input, ImageDataGenerator(no augmentation), flow_from_directory

    args:
        args.batch_size: int
        args.pretrain:
            boolean that indicates use pretrain model or not
        args.model_name:
            str if args.pretrain = True
        args.test_dir: test image directory which contains each class as separate folders
        args.img_size: target image size

    returns:
        data_generator:
            A DirectoryIterator yielding tuples of (x, y) where
                x -> a numpy array containing a batch of images with shape  (batch_size, *target_size, channels)
                y -> a numpy array of corresponding labels
    """

    preprocess_input = define_preprocess_input(args)

    ###############################
    # <ImageDataGenerator> class
    # goal: Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches)
    # :return:
    # Arguments:
    #   preprocessing_function:
    #       take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape,run after the image is resized and augmented
    ###############################
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    ####################
    # <flow_from_directory>
    # Goal: takes the path to a directory & generates batches of augmented data.
    #
    # input Args:
    #   directory: Path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
    #   classes: default None ->  automatically inferred from the subdirectory names under directory
    # return:
    #   A DirectoryIterator yielding tuples of (x, y) where
    #       x -> a numpy array containing a batch of images with shape  (batch_size, *target_size, channels)
    #       y -> a numpy array of corresponding labels
    ###################
    color_mode = 'grayscale' if args.channels == 1 else 'rgb'
    test_generator = test_datagen.flow_from_directory(
        directory=root_dir,
        # color_mode='grayscale', # 'rgb'
        target_size=(args.img_size, args.img_size),  # (height, width)
        # interpolation='nearest',
        color_mode=color_mode,
        batch_size=args.batch_size,
        class_mode='categorical'  # 2D one-hot encoded labels
    )
    return test_generator

def train_data_generator(root_dir, args):
    """ read train image data from directory using keras API
        including define preprocess_input, ImageDataGenerator(augmentation), flow_from_directory

        args:
            args.batch_size: int
            args.pretrain:
                boolean that indicates use pretrain model or not
            args.model_name:
                str if args.pretrain = True
            args.test_dir: test image directory which contains each class as separate folders
            args.img_size: target image size

        returns:
            data_generator:
                A DirectoryIterator yielding tuples of (x, y) where
                    x -> a numpy array containing a batch of images with shape  (batch_size, *target_size, channels)
                    y -> a numpy array of corresponding labels
    """
    preprocess_input = define_preprocess_input(args)

    ###############################
    # <ImageDataGenerator> class
    # goal: Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches)
    # :return:
    # Arguments:
    #   preprocessing_function:
    #       take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape,run after the image is resized and augmented
    ###############################
    train_datagen = image.ImageDataGenerator(
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        # rescale=1./255,
        preprocessing_function=preprocess_input,
        # rotation_range=30,
        # shear_range=0.1,
        # zoom_range=0.1,
        # vertical_flip=True,
        # horizontal_flip=True
    )

    ####################
    # <flow_from_directory>
    # Goal: takes the path to a directory & generates batches of augmented data.
    #
    # input Args:
    #   directory: Path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
    #   classes: default None ->  automatically inferred from the subdirectory names under directory
    # return:
    #   A DirectoryIterator yielding tuples of (x, y) where
    #       x -> a numpy array containing a batch of images with shape  (batch_size, *target_size, channels)
    #       y -> a numpy array of corresponding labels
    ###################
    color_mode = 'grayscale' if args.channels == 1 else 'rgb'
    train_generator = train_datagen.flow_from_directory(
        directory=args.train_dir,
        # color_mode='grayscale', # 'rgb'
        target_size=(args.img_size, args.img_size),  # (height, width)
        # interpolation='nearest',
        color_mode=color_mode,
        batch_size=args.batch_size,
        class_mode='categorical'  # 2D one-hot encoded labels
    )

    return train_generator


def init_pretrained_model(args):
    """ init model structure for transfer learning

    args:
        args.model_name
        args.img_size
        args.num_class
    return:
        model: keras.models.Model().compile()
    """

    MODELS = {
        "vgg16": vgg16.VGG16,
        "vgg19": vgg19.VGG19,
        "inception": inception_v3.InceptionV3,
        "xception": xception.Xception,
        "resnet50": resnet50.ResNet50
    }

    # init preprocess_input based on pre-trained model
    if args.model_name not in MODELS:
        raise AssertionError("model hasn't been pre-define yet, try: vgg16/vgg19/inception/xception/resnet50")

    print('loading the model and the pre-trained weights...')
    application = MODELS[args.model_name]
    base_model = application(
        include_top=False,
        weights='imagenet',  # weight model downloaded at .keras/models/
        # input_tensor=keras.layers.Input(shape=(224,224,3)), #custom input tensor
        input_shape=(args.img_size, args.img_size, 3)
    )

    # add additional layers (fc)
    x = base_model.output

    # in the future, can use diff args.model_architect in if
    if True:
        x = Flatten(name='top_flatten')(x)
        x = Dense(512, activation='relu', name='top_fc1')(x)
        x = Dropout(0.5, name='top_dropout')(x)
    predictions = Dense(args.num_class, activation='softmax', name='top_predictions')(x)

    # final model we will train
    # Model include all layers required in the computation of inputs and outputs
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # fix base_model layers, only train the additional layers
    for layer in base_model.layers:
        layer.trainable = False

    ######################
    # <Model.compile>
    # available loss: https://keras.io/losses/
    # available optimizers: https://keras.io/optimizers/
    ######################
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=["accuracy"])

    return model

def init_model_scratch(args):
    """ init model from scratch using keras functional API

    Here I used:
    input: 24*24*3
    conv1: 16*(3,3)
    pooling1: max
    conv2: 32*(3,3)
    pooling2
    conv3: 64*(3,3)
    pooling3
    fc1: 128
    dropout
    fc2: num_class
    """
    img_size = args.img_size
    channels = args.channels
    num_class = args.num_class
    inputs = Input(shape=(img_size, img_size, channels), name='input')
    conv1 = Conv2D(16, (3,3), padding='same', activation='relu', name='conv1')(inputs)
    pool1 = MaxPooling2D(name='pool1')(conv1)
    conv2 = Conv2D(32, (3,3), padding='same', activation='relu', name='conv2')(pool1)
    pool2 = MaxPooling2D(name='pool2')(conv2)
    conv3 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv3')(pool2)
    pool3 = MaxPooling2D(name='pool3')(conv3)
    flatten = Flatten(name='flatten')(pool3)
    fc1 = Dense(units=128, activation='relu', name='fc1')(flatten)
    dropout = Dropout(rate=0.5, name='dropout')(fc1)
    predictions = Dense(units=num_class, activation='softmax', name='prediction')(dropout)
    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


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


def plot_training(args, log_path):
    """plot training result from csv log file

    csv cols: epoch, acc, loss, val_acc, val_loss

    args:
        :log_path: csv_path
    return:
    """
    df = pd.read_csv(log_path)

    fig, ((ax0), (ax1)) = plt.subplots(nrows=2, ncols=1)
    ax0.plot(df['epoch'], df['acc'], label='training')
    if args.val_dir:
        ax0.plot(df['epoch'], df['val_acc'], label='validation')
    ax0.set_title('Accuracy')
    ax0.legend()

    ax1.plot(df['epoch'], df['loss'], label='training')
    if args.val_dir:
        ax1.plot(df['epoch'], df['val_loss'], label='validation')
    ax1.set_title('loss')
    ax1.legend()
    plt.show()

def evaluation_on_generator(data_dir, model, args):
    test_generator = test_data_generator(data_dir, args)
    # prediction = model.predict_generator(test_generator) # shape:(#sample, 4)
    # argmax
    score = model.evaluate_generator(test_generator)
    metrics = list(zip(model.metrics_names, score))  # [('loss',int),('acc', int)]
    print(metrics)

    return metrics

if __name__ == "__main__":
    ######### steps overview ########
    # 1. build data pipeline
    # 2. init CNN
    # 3. training
    # 4. evaluate on test data
    ########################################

    ######### pass params from script ########
    # set:
    #   args = init_args()
    # args instruction:
    # 1. train_dir/val_dir: specify training & testing data root path
    # 2. num_class
    # 3. target img_size
    # 4. channels: 1 for gray, 3 for color
    # 5. pretrain: True to use transfer learning, False to train from scratch
    # 6. model_name:
    # when pretrain = False
    #       • 'customized':
    #       • 'cifar10_wider':
    #       • 'cifar10_deeper':
    # when pretrain = True
    #       • 'vgg16'/'vgg19'/'inception'/'xception'/'resnet50'
    # 7. version_as_suffix: word to help differentiate experiment
    # 8. show_plot: whether to show training loss/acc plot at the end
    # • (optional) modify image augmentations to increase #samples in corresponding function in preprocess.py
    # • (optional) modify CNN structure in CNNs.py
    ########################################

    ######### pass params from terminal ########
    # set:
    #   args = parse_args()
    # run python train.py --train_dir train/ -val_dir test/
    ############################################

    class init_args(object):
        def __init__(self):
            self.callbacks_dir = '/Users/San/Projects/ImageClassifier/callbacks'
            self.train_dir = '/Users/San/Projects/ImageClassifier/data/CIFAR10/Train'
            self.val_dir = None
            self.test_dir = '/Users/San/Projects/ImageClassifier/data/CIFAR10/Test'
            self.num_class = 10
            self.img_size = 28  # resize target
            self.channels = 1
            self.pretrain = False # choose to use pretrained model or training from scratch
            self.model_name = 'customized' #customized/vgg16/vgg19/inception/xception/resnet50
            self.version_as_suffix = 'cifar_try1'
            self.batch_size = 64
            self.epochs = 2
            self.show_plot = False # plot loss/acc against epoch using CSVLogger data

    args = init_args()

    # build data
    train_generator = train_data_generator(args.train_dir, args)
    try:
        validation_generator = test_data_generator(args.val_dir, args)
    except:
        validation_generator = None

    # init CNN
    if args.pretrain:
        model = init_pretrained_model(args)
    elif args.model_name == 'customized' :
        model = init_model_scratch(args)
    else:
        raise ValueError('can not find matched model, try the following: customized/vgg16/vgg19/inception/xception/resnet50')

    model.summary()

    # training
    pretrain_model, tensorboard_dir, checkpoint_path, csv_path, time_path = train_on_generator(model, train_generator,validation_generator, args)

    if args.show_plot:
        plot_training(args, csv_path)

    ### evaluate on test data
    print('--------------------')
    print('predicting test dateset')
    saved_model = models.load_model(checkpoint_path)
    evaluation_on_generator(args, saved_model)
