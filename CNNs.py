from keras.applications import vgg16, vgg19, inception_v3, xception, resnet50
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import models, optimizers

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
        raise AssertionError("model hasn't been embedded yet, try: vgg16/vgg19/inception/xception/resnet50")

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
    args:
        args.img_size
        args.channels
        args.num_class
    return:
        model: keras.models.Model().compile()

    Here I used:
    input: 24*24*1
    conv1: 16*(3,3)
    pooling1: max
    conv2: 32*(3,3)
    pooling2
    conv3: 64*(3,3)
    pooling3
    fc1: 128
    dropout
    fc2: 10
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

def init_cifar10_wider(args):
    """
    input: 24*24*1
    conv1: 32*(3,3)
    pooling1: max
    conv2: 64*(3,3)
    pooling2
    conv3: 128*(3,3)
    fc1: 256
    dropout
    fc2: 128
    pred: 10
    """
    img_size = args.img_size
    channels = args.channels
    num_class = args.num_class

    inputs = Input(shape=(img_size, img_size, channels), name='input')
    conv1 = Conv2D(32, (3,3), padding='same', activation='relu', name='conv1')(inputs)
    pool1 = MaxPooling2D(name='pool1')(conv1)
    conv2 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv2')(pool1)
    pool2 = MaxPooling2D(name='pool2')(conv2)
    conv3 = Conv2D(128, (3,3), padding='same', activation='relu', name='conv3')(pool2)
    flatten = Flatten(name='flatten')(conv3)
    fc1 = Dense(units=512, activation='relu', name='fc1')(flatten)
    dropout = Dropout(rate=0.5, name='dropout')(fc1)
    fc2 = Dense(units=128, activation='relu', name='fc2')(dropout)
    predictions = Dense(units=num_class, activation='softmax', name='prediction')(fc2)
    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def init_cifar10_deeper(args):
    """

    """
    img_size = args.img_size
    channels = args.channels
    num_class = args.num_class

    inputs = Input(shape=(img_size, img_size, channels), name='input')
    conv1 = Conv2D(8, (3,3), padding='same', activation='relu', name='conv1')(inputs)
    conv2 = Conv2D(16, (3,3), padding='same', activation='relu', name='conv2')(conv1)
    pool1 = MaxPooling2D(name='pool1')(conv2)
    conv3 = Conv2D(32, (3,3), padding='same', activation='relu', name='conv3')(pool1)
    conv4 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv4')(conv3)
    pool2 = MaxPooling2D(name='pool2')(conv4)
    conv5 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv5')(pool2)
    flatten = Flatten(name='flatten')(conv5)
    fc1 = Dense(units=256, activation='relu', name='fc1')(flatten)
    dropout = Dropout(rate=0.5, name='dropout')(fc1)
    fc2 = Dense(units=128, activation='relu', name='fc2')(dropout)
    predictions = Dense(units=num_class, activation='softmax', name='prediction')(fc2)
    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def init_2CNN_model(args):
    """ CNN with 2 inputs/outputs independently
    args:
        args.img_size
        args.channels
        args.num_class
    :return:
        model: keras.models.Model().compile()
    """
    img_size = args.img_size
    channels = args.channels
    num_class = args.num_class

    # CNN 1
    a_inputs = Input(shape=(img_size, img_size, channels), name='a_input')
    a_conv1 = Conv2D(16, (3, 3), padding='same', activation='relu', name='a_conv1')(a_inputs)
    a_pool1 = MaxPooling2D(name='a_pool1')(a_conv1)
    a_conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', name='a_conv2')(a_pool1)
    a_pool2 = MaxPooling2D(name='a_pool2')(a_conv2)
    a_conv3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='a_conv3')(a_pool2)
    a_pool3 = MaxPooling2D(name='a_pool3')(a_conv3)
    a_flatten = Flatten(name='a_flatten')(a_pool3)
    a_fc1 = Dense(units=128, activation='relu', name='a_fc1')(a_flatten)
    a_dropout = Dropout(rate=0.5, name='a_dropout')(a_fc1)
    a_predictions = Dense(units=num_class, activation='softmax', name='a_pred')(a_dropout)

    # CNN 2
    b_inputs = Input(shape=(img_size, img_size, channels), name='b_input')
    b_conv1 = Conv2D(16, (3, 3), padding='same', activation='relu', name='b_conv1')(b_inputs)
    b_pool1 = MaxPooling2D(name='pool1')(b_conv1)
    b_conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', name='b_conv2')(b_pool1)
    b_pool2 = MaxPooling2D(name='b_pool2')(b_conv2)
    b_conv3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='b_conv3')(b_pool2)
    b_pool3 = MaxPooling2D(name='b_pool3')(b_conv3)
    b_flatten = Flatten(name='b_flatten')(b_pool3)
    b_fc1 = Dense(units=128, activation='relu', name='b_fc1')(b_flatten)
    b_dropout = Dropout(rate=0.5, name='b_dropout')(b_fc1)
    b_predictions = Dense(units=num_class, activation='softmax', name='b_pred')(b_dropout)

    losses = {
        'a_pred': 'categorical_crossentropy',
        'b_pred': 'categorical_crossentropy',
    }
    metrics = {
        'a_pred': 'accuracy',
        'b_pred': 'accuracy'
    }

    model = models.Model(inputs=[a_inputs, b_inputs], outputs=[a_predictions, b_predictions])
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses,
        loss_weights=[15,1],
        metrics=metrics
    )

    return model

def init_shared_model(args):
    """ CNN with 2 inputs/outputs shared first conv
    args:
        args.img_size
        args.channels
        args.num_class
    :return:
        model: keras.models.Model().compile()
    """
    img_size = args.img_size
    channels = args.channels
    num_class = args.num_class

    # shared nodes
    conv_1 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv1')

    # CNN 1
    a_inputs = Input(shape=(img_size, img_size, channels), name='a_input')
    a_conv1 = conv_1(a_inputs)
    a_pool1 = MaxPooling2D(name='a_pool1')(a_conv1)
    a_conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', name='a_conv2')(a_pool1)
    a_pool2 = MaxPooling2D(name='a_pool2')(a_conv2)
    a_conv3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='a_conv3')(a_pool2)
    a_pool3 = MaxPooling2D(name='a_pool3')(a_conv3)
    a_flatten = Flatten(name='a_flatten')(a_pool3)
    a_fc1 = Dense(units=128, activation='relu', name='a_fc1')(a_flatten)
    a_dropout = Dropout(rate=0.5, name='a_dropout')(a_fc1)
    a_predictions = Dense(units=num_class, activation='softmax', name='a_pred')(a_dropout)

    # CNN 2
    b_inputs = Input(shape=(img_size, img_size, channels), name='b_input')
    b_conv1 = conv_1(b_inputs)
    b_pool1 = MaxPooling2D(name='pool1')(b_conv1)
    b_conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', name='b_conv2')(b_pool1)
    b_pool2 = MaxPooling2D(name='b_pool2')(b_conv2)
    b_conv3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='b_conv3')(b_pool2)
    b_pool3 = MaxPooling2D(name='b_pool3')(b_conv3)
    b_flatten = Flatten(name='b_flatten')(b_pool3)
    b_fc1 = Dense(units=128, activation='relu', name='b_fc1')(b_flatten)
    b_dropout = Dropout(rate=0.5, name='b_dropout')(b_fc1)
    b_predictions = Dense(units=num_class, activation='softmax', name='b_pred')(b_dropout)

    losses = {
        'a_pred': 'categorical_crossentropy',
        'b_pred': 'categorical_crossentropy',
    }
    metrics = {
        'a_pred': 'accuracy',
        'b_pred': 'accuracy'
    }

    model = models.Model(inputs=[a_inputs, b_inputs], outputs=[a_predictions, b_predictions])
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses,
        loss_weights=[15,1],
        metrics=metrics
    )

    return model
