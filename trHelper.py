""" Helper for calling the tensor flow operations in creating the
convolution, max pool and fc layers"""

#Please note: The initial skeleton of this file was obtained from the Kadenze
# course "Creative Applications of Deep Learning w/ Tensorflow by Parag K. Mital"

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

def fc_layer(x, n_output, name=None, activation=None, reuse=None, dropout_dict = None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform = False))

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        if dropout_dict and dropout_dict['do_dropout']:
            print("applying dropout")
            h = tf.nn.dropout(h, dropout_dict['keep_prob'])

        return h, W

def conv2d(x, n_output,
           k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', activation=None, reuse=None, dropout_dict=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform = False))

        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding)

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

        if activation:
            h = activation(h)

        if dropout_dict and dropout_dict['do_dropout']:
            print("applying dropout")
            h = tf.nn.dropout(h, dropout_dict['keep_prob'])

    return h, W

def maxPool2d(x, k_h=2, k_w=2, d_h=2, d_w=2,
           padding='SAME', name='maxPool2d', reuse=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'maxPool2d', reuse=reuse):
        h = tf.nn.max_pool(x,
            name='h',
            ksize=[1,k_h,k_w,1],
            strides=[1, d_h, d_w, 1],
            padding=padding)
    return h
