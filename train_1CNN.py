import numpy as np
from keras import backend as K
from keras import models
from keras.utils import to_categorical

from utils.CNNs import init_pretrained_model, init_model_scratch, init_cifar10_wider, init_cifar10_deeper, \
    init_cifar2_new
from utils.plot import plot_training
from utils.preprocess import define_preprocess_func, data_pipeline
from utils.training import train

np.random.seed(2018)
K.clear_session()
K.set_image_dim_ordering('tf')



if __name__ == '__main__':
    class init_args(object):
        def __init__(self):
            self.callbacks_dir = '/Users/San/Projects/ImageClassifier/callbacks'
            self.train_dir = '/Users/San/Projects/ImageClassifier/data/cifar2/Train'
            self.val_dir = None
            self.test_dir = '/Users/San/Projects/ImageClassifier/data/cifar2/Test'
            self.num_class = 2
            self.img_size = 28  # resize target
            self.channels = 1
            self.pretrain = False # choose to use pretrained model or training from scratch
            self.model_name = 'cifar2_deeper' #customized/cifar10_wider/deeper/vgg16/vgg19/inception/xception/resnet50
            self.version_as_suffix = 'live'
            self.batch_size = 64
            self.epochs = 100
            self.show_plot = False # plot loss/acc against epoch using CSVLogger data

    args = init_args()

    # build data
    preprocess_input = define_preprocess_func(args)
    X_train, y_train = data_pipeline(args.train_dir, preprocess_input)
    y_train = to_categorical(y_train, args.num_class)

    # init CNN
    if args.pretrain:
        model = init_pretrained_model(args)
    elif args.model_name == 'customized':
        model = init_model_scratch(args)
    elif args.model_name == 'cifar10_wider':
        model = init_cifar10_wider(args)
    elif args.model_name == 'cifar10_deeper':
        model = init_cifar10_deeper(args)
    elif args.model_name == 'cifar2_deeper':
        model = init_cifar2_new(args)
    else:
        raise ValueError('can not find matched model, try the following: customized/vgg16/vgg19/inception/xception/resnet50')

    model.summary()

    # training
    pretrain_model, tensorboard_dir, checkpoint_path, csv_path, time_path = train(model, X_train, y_train, args)

    if args.show_plot:
        plot_training(args, csv_path)

    ### evaluate on test data
    print('--------------------')
    print('predicting test dateset')
    X_test, y_test = data_pipeline(args.test_dir, preprocess_input)
    y_test = to_categorical(y_test)
    saved_model = models.load_model(checkpoint_path)
    score = model.evaluate(X_test, y_test)
    print(list(zip(model.metrics_names, score)))