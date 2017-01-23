from keras.layers import Activation, MaxPooling2D, Convolution2D, Dropout, Input, merge, GlobalAveragePooling2D
from keras.models import Model

squeeze1 = "squeeze1x1"
expand1 = "expand1x1"
expand3 = "expand3x3"
relu = "relu_"


def fire_module(layer, fire_id, squeeze=16, expand=64, dim_ordering='th'):
    id = 'fire_node_' + str(fire_id) + '_'
    
    if dim_ordering is 'tf':
        axis = 3
    else:
        axis = 1
    
    layer = Convolution2D(squeeze, 1, 1, border_mode='valid', name=id + squeeze1)(layer)
    layer = Activation('relu', name=id + relu + squeeze1)(layer)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=id + expand1)(layer)
    left = Activation('relu', name=id + relu + expand1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=id + expand3)(layer)
    right = Activation('relu', name=id + relu + expand3)(right)

    x = merge([left, right], mode='concat', concat_axis=axis, name=id + 'concat')
    return x


def squeezenet(classes, dim_ordering='th', shape=(3,227,227)):
    # th -> (channel, width, height)
    # tf -> (width, height, channel)
    if dim_ordering is 'th':
        input_img = Input(shape=shape)
    elif dim_ordering is 'tf':
        input_img = Input(shape=shape)
    else:
        raise NotImplementedError("Theano and Tensorflow are only available")

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64, dim_ordering=dim_ordering)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128, dim_ordering=dim_ordering)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256, dim_ordering=dim_ordering)
    
    x = Dropout(0.5, name='drop9')(x)
    
    x = Convolution2D(classes, 1, 1, border_mode='valid', name='conv10')(x)
    x = Activation('relu', name='conv10_relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    out = Activation('softmax', name='loss')(x)
    
    model = Model(input=input_img, output=[out])
    return model