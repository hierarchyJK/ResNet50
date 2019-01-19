# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2019-01-16 19:38:57
@month:一月
"""
import numpy as np
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydoc
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n,_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in CONV layers of main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/chracter, used to name the layers, depending on their position in the network
    :return:
    X: output of the identity block,tensor of shape(n_H, n_W, n_C)
    """
    #defining name basis
    conv_name_base = 'res_identity_block' + str(stage) + block + '_branch'
    bn_name_base = 'bn_identity_block' + str(stage) + block + '_branch'

    #retrieve filters
    F1, F2, F3 = filters

    #save the input value, you'll need this later to add back to the main path
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to the main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block_test():
    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3,4,4,6)
        A = identity_block(A_prev, f=2, filters = [2, 4, 6], stage=1, block='a')
        test.run(tf.global_variables_initializer())
        out = test.run([A],feed_dict={A_prev: X, K.learning_phase():0})
        print("identity_block'out = " + str(out[0][1][1][0]))
identity_block_test()


def convolution_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined
    :param X: input tensor of shape(m, n_H_prev, n_W_prev, n_C_prev)
    :param f:integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block:string/character, used to name the layers, depending on their position in the network
    :param s: Integer, specifying the stride to be used
    :return:X:output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    #defining name basis
    conv_name_base = 'res_convolution_block' + str(stage) + block + '_branch'
    bn_name_base = 'bn_convolution_block' + str(stage) + block + '_branch'

    #Retrieve Filters
    F1, F2, F3 = filters

    #Save the input value
    X_shortcut = X

    ###### Main Path ######
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name = conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    ##### Shortcut path ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final Step:Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

def convolution_block_test():
    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3,4,4,6])
        X = np.random.rand(3,4,4,6)
        A = convolution_block(A_prev, f = 2, filters=[2,4,6], stage=1, block='a')
        test.run(tf.global_variables_initializer())
        out = test.run([A],feed_dict={A_prev:X, K.learning_phase():0})
        print("convolution_block'out = " + str(out[0][1][1][0]))
convolution_block_test()


"""    下面开始Building一个50层的ResNet model"""
def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BN -> RELU -> MAXPOOL
    -> CONVBLOCK -> IDBLOCK*2
    -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5
    -> CONVBLOCK -> IDBLOCK*2
    -> AVGPOOL -> TOPLAYER
    :param input_shape: shape of the images of the dataset
    :param classes: integer, number of classes
    :return:
    model -- a Model() instance in Keras
    """
    # Defines the input a s tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero_Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2,2), name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolution_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolution_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolution_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolution_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # FLATTEN
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
#####创建模型实体
model = ResNet50(input_shape=(64, 64, 3), classes = 6)
#####编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
######训练模型
model.fit(X_train, Y_train, epochs=20, batch_size=32)
model.save('ResNet50.h5')######保存模型
"""
补充：保存模型之后自然我们需要加载模型
from keras.models import load_model
model = load_dataset("ResNet50.h5")
模型加载后可以接着训练，而不需要从头开始。

"""
######评估模型
preds = model.evaluate(X_test, Y_test)
print("误差值Test Loss = " + str(preds[0]))
print("精确度Test Accuracy" + str(preds[1]))


















