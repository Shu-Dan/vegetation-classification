from tensorflow.keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPool2D, Dropout, concatenate, merge, UpSampling2D,Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from nets.resnet50 import get_resnet50_encoder
from nets.vgg16 import VGG16


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
    # return 0.5 * categorical_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


#############################
# Unet++ conv and upsampling #
#############################
def conv_drop(inputs, filters):
    conv1 = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal')(inputs)
    # drop1 = Dropout(rate=0.5)(conv1)
    conv2 = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal')(conv1)
    # drop2 = Dropout(rate=0.5)(conv2)
    return conv2


def upsampling(inputs, filters):
    up = UpSampling2D(size=(2, 2))(inputs)
    conv = Conv2D(filters=filters, kernel_size=2, activation='relu', padding='same',
                  kernel_initializer='he_normal')(up)
    return conv


###############################
# Unet++                      #
###############################
def Unett(input_size,n_class, filters=(32,64, 128, 256, 512), re_shape=False):
    inputs = Input(input_size)

    # conv0_0, conv1_0, conv2_0, conv3_0, conv4_0 = get_resnet50_encoder(inputs, downsample_factor=16)
    conv0_0, conv1_0, conv2_0, conv3_0, conv4_0=VGG16(inputs)


    ## l1
    conv0_0 = conv_drop(inputs=inputs, filters=filters[0])
    pool00_10 = MaxPool2D(pool_size=(2, 2))(conv0_0)
    # conv1_0 = conv_drop(inputs=pool00_10, filters=filters[1])
    up10_01 = upsampling(inputs=conv1_0, filters=filters[0])
    concat1_1 = concatenate([up10_01, conv0_0], axis=3)
    conv0_1 = conv_drop(inputs=concat1_1, filters=filters[0])
    ## l2
    pool10_20 = MaxPool2D(pool_size=(2, 2))(conv1_0)
    # conv2_0 = conv_drop(inputs=pool10_20, filters=filters[2])
    up20_11 = upsampling(inputs=conv2_0, filters=filters[1])
    concat2_1 = concatenate([up20_11, conv1_0], axis=3)
    conv1_1 = conv_drop(inputs=concat2_1, filters=filters[1])
    up11_02 = upsampling(inputs=conv1_1, filters=filters[0])
    concat2_2 = concatenate([up11_02, conv0_0, conv0_1], axis=3)
    conv0_2 = conv_drop(inputs=concat2_2, filters=filters[0])
    ##l3
    pool20_30 = MaxPool2D(pool_size=(2, 2))(conv2_0)
    # conv3_0 = conv_drop(inputs=pool20_30, filters=filters[3])
    up30_21 = upsampling(inputs=conv3_0, filters=filters[2])
    concat3_1 = concatenate([up30_21, conv2_0], axis=3)
    conv2_1 = conv_drop(inputs=concat3_1, filters=filters[2])
    up21_12 = upsampling(inputs=conv2_1, filters=filters[1])
    concat3_2 = concatenate([up21_12, conv1_0, conv1_1], axis=3)
    conv1_2 = conv_drop(inputs=concat3_2, filters=filters[1])
    up12_03 = upsampling(inputs=conv1_2, filters=filters[0])
    concat3_3 = concatenate([up12_03, conv0_0, conv0_1, conv0_2], axis=3)
    conv0_3 = conv_drop(inputs=concat3_3, filters=filters[0])
    ## l4
    pool30_40 = MaxPool2D(pool_size=(2, 2))(conv3_0)
    # conv4_0 = conv_drop(inputs=pool30_40, filters=filters[4])
    up40_31 = upsampling(inputs=conv4_0, filters=filters[3])
    concat4_1 = concatenate([up40_31, conv3_0], axis=3)
    conv3_1 = conv_drop(inputs=concat4_1, filters=filters[3])
    up31_22 = upsampling(inputs=conv3_1, filters=filters[2])
    concat4_2 = concatenate([up31_22, conv2_0, conv2_1], axis=3)
    conv2_2 = conv_drop(inputs=concat4_2, filters=filters[2])
    up22_13 = upsampling(inputs=conv2_2, filters=filters[1])
    concat4_3 = concatenate([up22_13, conv1_0, conv1_1, conv1_2], axis=3)
    conv1_3 = conv_drop(inputs=concat4_3, filters=filters[1])
    up13_04 = upsampling(inputs=conv1_3, filters=filters[0])
    concat4_4 = concatenate([up13_04, conv0_0, conv0_1, conv0_2, conv0_3], axis=3)
    conv0_4 = conv_drop(inputs=concat4_4, filters=filters[0])
    ## output
    l1_conv_out = Conv2D(filters=n_class, kernel_size=1, padding='same', kernel_initializer='he_normal')(conv0_1)
    l2_conv_out = Conv2D(filters=n_class, kernel_size=1, padding='same', kernel_initializer='he_normal')(conv0_2)
    l3_conv_out = Conv2D(filters=n_class, kernel_size=1, padding='same', kernel_initializer='he_normal')(conv0_3)
    l4_conv_out = Conv2D(filters=n_class, kernel_size=1, padding='same', kernel_initializer='he_normal')(conv0_4)

    if re_shape == True:
        l1_conv_out = tf.Reshape((input_size[0] * input_size[1], n_class))(l1_conv_out)
        l2_conv_out = tf.Reshape((input_size[0] * input_size[1], n_class))(l2_conv_out)
        l3_conv_out = tf.Reshape((input_size[0] * input_size[1], n_class))(l3_conv_out)
        l4_conv_out = tf.Reshape((input_size[0] * input_size[1], n_class))(l4_conv_out)

    l1_out = Activation('sigmoid', name='l1_out')(l1_conv_out)
    l2_out = Activation('sigmoid', name='l2_out')(l2_conv_out)
    l3_out = Activation('sigmoid', name='l3_out')(l3_conv_out)
    l4_out = Activation('sigmoid', name='l4_out')(l4_conv_out)

    model = Model(inputs=inputs, outputs=[l1_out, l2_out, l3_out, l4_out])
    # model = Model(input=inputs, output=l4_out)
    model.summary()
    losses = {
        'l1_out': bce_dice_loss,
        'l2_out': bce_dice_loss,
        'l3_out': bce_dice_loss,
        'l4_out': bce_dice_loss,
    }
    model.compile(optimizer=Adam(lr=1e-4), loss=losses, metrics=['accuracy'])


    return model
