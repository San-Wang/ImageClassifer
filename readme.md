# Image Classifier

MOTIVATION
==========  
ML challenge

CHECKBOX  
========  
```
Content included in this repo:  
|--Read data  
    |-- preprocess  
        |-- √resize  
        |-- √normalize  
        |-- √augmentation  
|-- Training  
    |--√build network  
    |--√save callbacks  
    |--√transfer learning  
    |--grid search tuning 
|--Evaluation   
    |--√plot loss/accuracy    
    |--predict  
        a. single image  
        √b. batch images  
    |--√confusion matrix  
```  

Future add on
-------------  
[ ] 
[ ] automating tensorboard viz for deeper layers  
[ ] https://js.tensorflow.org/  
note: train and deploy model in browser  


DATASET
==============  
* data structure  
```
Cells  
Train: 2000 28*28 gray images evenly in 4 classes  
Test: 800 28*28 gray images evenly in 4 classes  
|--Train/Test  
    |--0/ single cells.png  
    |--1/ double.png  
    |--2/ triple.png  
    |--3/ four.png  
    
CIFAR10  
Train: 10,000 28*28 gray images evenly in 10 classes  
Test: 1,000 28*28 gray images evenly in 10 classes  
|--Train/Test  
    |--0/ airplane.png  
    |--1/ automobile.png  
    |--2/ bird.png  
    |--3/ cat.png  
    |--4/ deer.png  
    |--5/ dog.png  
    |--6/ frog.png  
    |--7/ horse.png  
    |--8/ ship.png  
    |--9/ truck.png  
    
MNIST  
Train: 10,000 28*28 gray images evenly in 10 classes    
Test: 1,000 28*28 gray images evenly in 10 classes 
|--Train/Test  
    |--0/ 0.png  
    |--1/ 1.png  
    |--2/ 2.png  
    |--3/ 3.png  
    |--4/ 4.png  
    |--5/ 5.png  
    |--6/ 6.png  
    |--7/ 7.png  
    |--8/ 8.png  
    |--9/ 9.png  
```
    
* data cleaning log  


USE INSTRUCTION  
===============  
    
environment  
-----------  
python3.6  

Scripts  
------  
modules:  
proprocess.py: various data reading pipelines  
CNNs.py: various pre-defined CNN structure  
training.py: define training process   
metrics.py: functions that evaluation model performance  
plot.py: plot functions  

main scripts:  
train_*.py

    ##### steps in main script overview ########
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
    # 6. model_name: indicator which CNN structure to use  
    # when pretrain = False
    #       • 'customized': use customized CNN
    #       • 'cifar10_wider': 
    #       • 'cifar10_deeper':
    # when pretrain = True
    #       • 'vgg16'/'vgg19'/'inception'/'xception'/'resnet50': use transfer learning(only work when channels = 3)
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

REFERENCE
=========
https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html  
https://www.cs.toronto.edu/~kriz/cifar.html  
http://alexlenail.me/NN-SVG/LeNet.html  


Experiment results:  
==================  

Q1:  

| model name| time (mins) | test acc |  
|-----------|------|----------|  
| customized_4_28_50_64_cell.h5 | 1.53 | 0.95625 |  

Q2 (a)   

| model name| time (mins) | test acc |  loss weight | note |  
|-----------|-------------|----------|--------------|------|  
| 2CNN_10_28_100_64_try2.h5 | 27.81 | ('a_pred_acc', 0.45), ('b_pred_acc', 0.975) | 30:1 | loss doesn't improve around a_pred_acc: 0.8629 |  
| 2CNN_10_28_100_64_try2.h5 | 26.33 | ('a_pred_acc', 0.453), ('b_pred_acc', 0.974) | 15: 1 | loss doesn't improve start epoch 71, around a_pred_acc: 0.77 |  
| 2CNN_10_28_100_64_15:1.h5 | 30.20 | ('a_pred_acc', 0.411), ('b_pred_acc', 0.978) | 15:1 |  |  


Q2 (b)  

| model name| time (mins) | test acc |  loss weight |  
|-----------|-------------|----------|--------------|  
| shared_10_28_100_64_try.h5 | 26.75 | ('a_pred_acc', 0.492), ('b_pred_acc', 0.974) | 15: 1 |  
| shared_10_28_100_64_15:1.h5 | 24.21 | ('a_pred_acc', 0.453), ('b_pred_acc', 0.971) | 15:1 |  


only classify cifar10  

| model name| time (mins) | test acc | note |     
|-----------|-------------|----------|------|  
| customized_10_28_100_64_shuffle.h5 | 14.09 | 0.509 | |  
| cifar10_deeper_10_28_100_64_try1.h5 | 37.2 | 0.516 | |  
| cifar10_10_28_100_64_try.h5 | 35.57 | 0.518 | wider CNN, acc quickly reach high but finally result not high|  
  