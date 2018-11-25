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
    plt.show()

def plot_time(time_path):
    df = pd.read_csv(time_path, parse_dates=['start_datetime','end_datetime'])

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(df['epoch'], df['start_datetime'], '-o')
    ax.set_title('training timeline')
    fig.autofmt_xdate()
    plt.show()



if __name__ == '__main__':
    plot_time('/Users/San/DataSciences/Capstone/ImageClassifier/callbacks/pretrained_vgg16_2_80_3_2_testTime_time.csv')
