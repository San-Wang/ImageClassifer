from keras.preprocessing import image

from utils.CNNs import init_pretrained_model, init_model_scratch, init_cifar10_wider, init_cifar10_deeper
from utils.metrics import evaluation_on_generator
from utils.plot import plot_training
from utils.preprocess import define_preprocess_func
from utils.training import train_on_generator

K.clear_session()
K.set_image_dim_ordering('tf')

# over-write train_data_generator to add augmentation

def train_data_generator(root_dir, args):
    """ read train image data from directory using keras API
    """
    #preprocess_input = define_preprocess_func(args)

    train_datagen = image.ImageDataGenerator(
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1. / 255,
        #preprocessing_function=preprocess_input,
        rotation_range=30,
        shear_range=0.1,
        # zoom_range=0.1,
        # vertical_flip=True,
        # horizontal_flip=True
    )

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


if __name__ == '__main__':
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
            self.model_name = 'customized' #customized/cifar10_wider/deeper/vgg16/vgg19/inception/xception/resnet50
            self.version_as_suffix = 'aug_try'
            self.batch_size = 64
            self.epochs = 200
            self.show_plot = False # plot loss/acc against epoch using CSVLogger data

    args = init_args()

    # build data
    preprocess_input = define_preprocess_func(args)
    train_generator = train_data_generator(args.train_dir, args)

    # init CNN
    if args.pretrain:
        model = init_pretrained_model(args)
    elif args.model_name == 'customized':
        model = init_model_scratch(args)
    elif args.model_name == 'cifar10_wider':
        model = init_cifar10_wider(args)
    elif args.model_name == 'cifar10_deeper':
        model = init_cifar10_deeper(args)
    else:
        raise ValueError('can not find matched model, try the following: customized/vgg16/vgg19/inception/xception/resnet50')

    model.summary()

    # training
    pretrain_model, tensorboard_dir, checkpoint_path, csv_path, time_path = train_on_generator(model, train_generator, None, args)

    if args.show_plot:
        plot_training(args, csv_path)

    ### evaluate on test data
    print('--------------------')
    print('predicting test dateset')
    evaluation_on_generator(args.test_dir, pretrain_model, args)