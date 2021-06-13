# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:32:35 2018

@author: Kashish
"""

# Import Libraries
import tensorflow as tf
import numpy as np
from glob import glob
import os
import cv2


# Build a class for model
class Model:

    def __init__(self):
        pass

    def new_conv_layer(self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(bottom, w, [1, stride, stride, 1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))
        return bias

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1,
                         name=None):
        with tf.compat.v1.variable_scope(name):
            W = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-2], initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose(bottom, W, output_shape, [1, stride, stride, 1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))
        return bias

    def new_fc_layer(self, bottom, output_size, name):
        shape = bottom.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(bottom, [-1, dim])
        input_size = dim
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable("W", shape=[input_size, output_size],
                                initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=[output_size], initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
        return fc

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak * bottom, bottom)

    def build_reconstruction(self, images, is_train):
        with tf.compat.v1.variable_scope('GEN'):
            # Convolution Layer  
            # VGG Model
            conv1_1 = self.new_conv_layer(images, [3, 3, 3, 32], stride=1, name="conv1_1")
            conv1_1 = tf.nn.elu(conv1_1)
            conv1_2 = self.new_conv_layer(conv1_1, [3, 3, 32, 32], stride=1, name="conv1_2")
            conv1_2 = tf.nn.elu(conv1_2)
            # MaxPooling-1
            conv1_stride = self.new_conv_layer(conv1_2, [3, 3, 32, 32], stride=2, name="conv1_stride")

            conv2_1 = self.new_conv_layer(conv1_stride, [3, 3, 32, 64], stride=1, name="conv2_1")
            conv2_1 = tf.nn.elu(conv2_1)
            conv2_2 = self.new_conv_layer(conv2_1, [3, 3, 64, 64], stride=1, name="conv2_2")
            conv2_2 = tf.nn.elu(conv2_2)
            # MaxPooling-2
            conv2_stride = self.new_conv_layer(conv2_2, [3, 3, 64, 64], stride=2, name="conv2_stride")

            conv3_1 = self.new_conv_layer(conv2_stride, [3, 3, 64, 128], stride=1, name="conv3_1")
            conv3_1 = tf.nn.elu(conv3_1)
            conv3_2 = self.new_conv_layer(conv3_1, [3, 3, 128, 128], stride=1, name="conv3_2")
            conv3_2 = tf.nn.elu(conv3_2)
            conv3_3 = self.new_conv_layer(conv3_2, [3, 3, 128, 128], stride=1, name="conv3_3")
            conv3_3 = tf.nn.elu(conv3_3)
            conv3_4 = self.new_conv_layer(conv3_3, [3, 3, 128, 128], stride=1, name="conv3_4")
            conv3_4 = tf.nn.elu(conv3_4)
            # MaxPooling-3
            conv3_stride = self.new_conv_layer(conv3_4, [3, 3, 128, 128], stride=2,
                                               name="conv3_stride")  # Final feature map (temporary)

            conv4_stride = self.new_conv_layer(conv3_stride, [3, 3, 128, 128], stride=2, name="conv4_stride")  # 16 -> 8
            conv4_stride = tf.nn.elu(conv4_stride)

            conv5_stride = self.new_conv_layer(conv4_stride, [3, 3, 128, 128], stride=2, name="conv5_stride")  # 8 -> 4
            conv5_stride = tf.nn.elu(conv5_stride)

            conv6_stride = self.new_conv_layer(conv5_stride, [3, 3, 128, 128], stride=2, name="conv6_stride")  # 4 -> 1
            conv6_stride = tf.nn.elu(conv6_stride)

            # Deconvolution Layer with Skip Connections
            deconv5_fs = self.new_deconv_layer(conv6_stride, [3, 3, 128, 128], conv5_stride.get_shape().as_list(),
                                               stride=2, name="deconv5_fs")
            debn5_fs = tf.nn.elu(deconv5_fs)

            skip5 = tf.concat([debn5_fs, conv5_stride], 3)
            channels5 = skip5.get_shape().as_list()[3]

            deconv4_fs = self.new_deconv_layer(skip5, [3, 3, 128, channels5], conv4_stride.get_shape().as_list(),
                                               stride=2, name="deconv4_fs")
            debn4_fs = tf.nn.elu(deconv4_fs)

            skip4 = tf.concat([debn4_fs, conv4_stride], 3)
            channels4 = skip4.get_shape().as_list()[3]

            deconv3_fs = self.new_deconv_layer(skip4, [3, 3, 128, channels4], conv3_stride.get_shape().as_list(),
                                               stride=2, name="deconv3_fs")
            debn3_fs = tf.nn.elu(deconv3_fs)

            skip3 = tf.concat([debn3_fs, conv3_stride], 3)
            channels3 = skip3.get_shape().as_list()[3]

            deconv2_fs = self.new_deconv_layer(skip3, [3, 3, 64, channels3], conv2_stride.get_shape().as_list(),
                                               stride=2, name="deconv2_fs")
            debn2_fs = tf.nn.elu(deconv2_fs)

            skip2 = tf.concat([debn2_fs, conv2_stride], 3)
            channels2 = skip2.get_shape().as_list()[3]

            deconv1_fs = self.new_deconv_layer(skip2, [3, 3, 32, channels2], conv1_stride.get_shape().as_list(),
                                               stride=2, name="deconv1_fs")
            debn1_fs = tf.nn.elu(deconv1_fs)

            skip1 = tf.concat([debn1_fs, conv1_stride], 3)
            channels1 = skip1.get_shape().as_list()[3]

            recon = self.new_deconv_layer(skip1, [3, 3, 3, channels1], images.get_shape().as_list(), stride=2,
                                          name="recon")

        return recon


# Global Variables

pen_size = 1
img_idx = 0
drawing = False
ix, iy = -1, -1
vis_size = 320
blank_size = 20


# Utilities

def nothing(x):
    pass


def draw(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (255, 255, 255), pen_size)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), pen_size)


def masking(image):
    mask = (np.array(image[:, :, 0]) == 255) & (np.array(image[:, :, 1]) == 255) & (np.array(image[:, :, 2]) == 255)
    mask = np.dstack([mask, mask, mask])
    return (True ^ mask) * np.array(image)


# Image Pre-Processing

img_paths = []
img_paths.extend(sorted(glob(os.path.join("../Test Image/", '*.bmp'))))
img_ori = cv2.imread(img_paths[img_idx]) / 255.
img = img_ori
empty = np.zeros((vis_size, vis_size, 3))
blank = np.zeros((vis_size, blank_size, 3)) + 1
text_region = np.zeros((blank_size, 2 * vis_size + blank_size, 3)) + 1.
recon_img = empty

cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Window', draw)

# Tensorflow Session

sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()

pretrained_model_path = "../Pretrained Weight/model_mscoco"

is_train = tf.compat.v1.placeholder(tf.bool)
images_tf = tf.compat.v1.placeholder(tf.float32, shape=[1, vis_size, vis_size, 3], name="images")

model = Model()
reconstruction_ori = model.build_reconstruction(images_tf, is_train)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
saver.restore(sess, pretrained_model_path)

# Main Function Loop

while 1:
    view = np.hstack((img, blank, recon_img[:, :, [2, 1, 0]]))
    window = np.vstack((view, text_region))
    cv2.imshow('Window', window)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == 99:
        masked_input = masking(img)
        masked_input = masked_input[:, :, [2, 1, 0]]
        shape3d = np.array(masked_input).shape
        print(shape3d)
        model_input = np.array(masked_input).reshape((1, shape3d[0], shape3d[1], shape3d[2]))
        print(model_input.shape)
        model_output = sess.run(reconstruction_ori, feed_dict={images_tf: model_input, is_train: False})
        recon_img = np.array(model_output)[0, :, :, :].astype(float)
        cv2.imwrite(os.path.join("../Result", img_paths[img_idx][21:35]), ((recon_img[:, :, [2, 1, 0]]) * 255))
        cv2.imwrite(os.path.join("../Input", img_paths[img_idx][21:35]), (img * 255))

cv2.destroyAllWindows()
