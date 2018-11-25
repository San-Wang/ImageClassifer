from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize

import keras
from keras.preprocessing.image import array_to_img, img_to_array
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


def imgGen(img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,
           preprocess_fcn=None, batch_size=9):
    datagen = ImageDataGenerator(
        zca_whitening=zca,
        rotation_range=rotation,
        width_shift_range=w_shift,
        height_shift_range=h_shift,
        shear_range=shear,
        zoom_range=zoom,
        fill_mode='nearest',
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        preprocessing_function=preprocess_fcn,
        data_format=K.image_data_format())

    #datagen.fit(img)

    i = 0
    for img_batch in datagen.flow_from_directory(img_path, batch_size=9, shuffle=False):
        for img in img_batch:
            plt.subplot(330 + 1 + i)
            if img.shape[-1] == 1:
                img = array_to_img(img)
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            i = i + 1
        if i >= batch_size:
            break
    plt.show()

def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

img_path = '/Users/San/DataSciences/Capstone/data/2class/00001/00000_00000.ppm'
#img_path = '/Users/San/Projects/ImageClassifier/data/Cells/Test/0/Image00000.png'

img = imread(img_path).astype('float32')
img = keras.preprocessing.image.img_to_array(img)
img_path = '/Users/San/DataSciences/Capstone/data/2class'
img /= 255
h_dim = np.shape(img)[0]
w_dim = np.shape(img)[1]
num_channel = np.shape(img)[2]
img = img.reshape(1, h_dim, w_dim, num_channel)
imgGen(img_path, rotation=30, h_shift=0.5, preprocess_fcn = HE)
