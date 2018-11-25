import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    img = np.multiply(img, 1.0 / 255.0)

    return img

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')

    for fields in classes:
        index = classes.index(fields)
        print('Reading file {} (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*.png')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            #image = image[:,:,(2,1,0)] # swapping BGR to RGB
            image = transform_img(image, image_size, image_size)

            image = image.astype(np.float32)
            images.append(image)
            label = np.zeros(len(classes)) # shape = (len, )
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl) # get path last element: ###_###.jpg
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images) # change list to ndarray. shape = (###,80,80,3)
    labels = np.array(labels) # shape = (###, len(class))
    img_names = np.array(img_names) # shape = (###, )
    cls = np.array(cls)
    print('Have {} categories of images in total'.format(len(classes)))

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def num_examples(self):
      return self._num_examples

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, test_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  # print('Before shuffling:\n First 5 images names :{}'.format(img_names[:5]))
  print('shuffling images')
  #np.random.seed(3)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
  #print('After shuffling:\n First 5 images names :{}'.format(img_names[:5]))


  if isinstance(test_size, float):
    test_size = int(test_size * images.shape[0])

  test_images = images[:test_size]
  test_labels = labels[:test_size]
  test_img_names = img_names[:test_size]
  test_cls = cls[:test_size]

  train_images = images[test_size:]
  train_labels = labels[test_size:]
  train_img_names = img_names[test_size:]
  train_cls = cls[test_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)

  return data_sets