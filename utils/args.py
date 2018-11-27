import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--callbacks_dir", type=str, default='./callbacks', help="directory path to save callbacks files")
    ap.add_argument("--train_dir", type=str, required=True, help="(required) the train data directory")
    ap.add_argument("--val_dir", type=str, default=None, help="the validation data directory")
    ap.add_argument("test_dir", type=str, default=None, help="test data directory")
    ap.add_argument("--num_class", type=int, required=True, help="number of classes to classify")
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
