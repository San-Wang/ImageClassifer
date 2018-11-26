import matplotlib.pyplot as plt
import pandas as pd

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
    #plt.show()

def plot_2CNN_training(args, log_path):
    """plot training result from csv log file

    csv cols: epoch, acc, loss, val_acc, val_loss

    args:
        :log_path: csv_path
    return:
    """
    df = pd.read_csv(log_path)

    fig, ((ax0), (ax1)) = plt.subplots(nrows=2, ncols=1)
    ax0.plot(df['epoch'], df['a_pred_acc'], label='CIFAR')
    ax0.plot(df['epoch'], df['b_pred_acc'], label='MNIST')
    ax0.set_title('Accuracy')
    ax0.legend()

    ax1.plot(df['epoch'], df['a_pred_loss'], label='CIFAR')
    ax1.plot(df['epoch'], df['b_pred_loss'], label='MNIST')
    #ax1.plot(df['epoch'], df['loss'], label='loss')
    ax1.set_title('Loss')
    ax1.legend()
    #plt.show()

def plot_time(time_path):
    df = pd.read_csv(time_path, parse_dates=['start_datetime','end_datetime'])

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(df['epoch'], df['start_datetime'], '-o')
    ax.set_title('training timeline')
    fig.autofmt_xdate()
    #plt.show()



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
            self.version_as_suffix = 'shuffle'
            self.batch_size = 64
            self.epochs = 200
            self.show_plot = False

    class init_2CNN_args(object):
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
    #args = init_2CNN_args()

    model_name = 'customized_10_28_100_64_shuffle'

    log_path = '/Users/San/Projects/ImageClassifier/callbacks/{}.log'.format(model_name)
    fig_path = '/Users/San/Projects/ImageClassifier/Present/{}_loss.png'.format(model_name)

    plot_training(args, log_path)
    #plot_2CNN_training(args, log_path)
    plt.savefig(fig_path, bbox_inches='tight')

