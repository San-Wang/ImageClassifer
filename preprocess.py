import os
import glob
import cv2
import PIL
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.applications import vgg16, vgg19, inception_v3, xception, resnet50, imagenet_utils
from keras.preprocessing import image

def define_preprocess_func(args):
    """ define preprocess_input from args param for generator
    args:
        :args.pretrain: boolean
        :args.channels: int
        :args.model_name: str
    :return:
        :preprocess_input: The function should take one argument:
                           one image (Numpy tensor with rank 3),
                           and should output a Numpy tensor with the same shape.
    """
    MODELS = {
        "vgg16": vgg16.VGG16,
        "vgg19": vgg19.VGG19,
        "inception": inception_v3.InceptionV3,
        "xception": xception.Xception,
        "resnet50": resnet50.ResNet50
    }

    # when use customized structure
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

    elif args.channels == 1:
        def preprocess_input(x):
            img = image.img_to_array(x)
            # resize
            img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            img = image.img_to_array(img)
            # normalization
            img /= 225.0
            img = image.array_to_img(img)
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


def transform_img(img, args):
    """ transform ndarray image

    modify any transformation that share across training & testing

    args:
        :img: ndarray, grayscale->[size, size], rgb->[*,*,3]
    :return: same size ndarray
    """
    # resize
    img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
    # normalized
    img /= 255.0
    return img


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

    preprocess_input = define_preprocess_func(args)

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
            args.channels: 1 or 3 -> decide color_mode & preprocess
            args.img_size: target image size
            args.pretrain:
                boolean that indicates use pretrain model or not
            args.model_name:
                str -> decide preprocess
            args.test_dir: test image directory which contains each class as separate folders

        returns:
            data_generator:
                A DirectoryIterator yielding tuples of (x, y) where
                    x -> a numpy array containing a batch of images with shape  (batch_size, *target_size, channels)
                    y -> a numpy array of corresponding labels
    """
    preprocess_input = define_preprocess_func(args)

    ###############################
    # <ImageDataGenerator> class
    # goal: Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches)
    # :return:
    # Arguments:
    #   preprocessing_function:
    #       take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape,run after the image is resized and augmented
    ###############################
    # modify augmentation here
    train_datagen = image.ImageDataGenerator(
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        # rescale=1. / 255,
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


def get_class_from_folder(img_path):
    """ get class from parents' folder
    args:
        :param img_path: str
        eg: .../class_folder/image_name.png
    :return: int
    """
    return int(img_path.split('/')[-2])


def data_pipeline(root_dir, transfrom_img):
    """ read images from folder to ndarray

    args:
        :root_dir: str, the path that contain images in sub folders(indicate class)
        :transfrom_img: use preprocess.define_preprocess_func

    return:
        :X: ndarray -> image data shape: [#sample, img_size,img_size, channels]
        :y: adarray -> label shape = [#sample,]

    root_dir example:
    root_dir
        |--0/
            |--filename.png
        |--1/

    approach1:
        use flow_from_dictory
    approach2:
        vanilla way using numpy
    """

    X = []
    y = []
    image_paths = glob.glob(os.path.join(root_dir, '*/*'))
    for image_path in image_paths:
        img = mpimg.imread(image_path)
        #cv2.imread need to swapp BGR to RGB: img[:, :, (2, 1, 0)]
        img = transfrom_img(img)
        img = image.img_to_array(img)
        X.append(img)
        label = get_class_from_folder(image_path)
        y.append(label)

    X = np.array(X)  # change list to ndarray. shape = (###,80,80,3)
    y = np.array(y)  # shape = (###,)
    X, y = shuffle(X, y)
    print('In directory{}\nclass, #images:'.format(root_dir))
    print(pd.Series(y).value_counts())

    return X, y

def img_pipeline(img_path, args):

    X = []
    y = []
    img = mpimg.imread(img_path)
    img = transform_img(img, args)
    img = image.img_to_array(img)
    img = np.expand_dims(img, 0)
    label = get_class_from_folder(img_path)
    X.append(img)
    y.append(label)
    return X, y

