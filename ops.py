import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import pprint
from demod import conv2ddemod
pp = pprint.PrettyPrinter()
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])
import tensorflow.contrib as tf_contrib
# import tensorflow.contrib as tf_contrib
# weight_init = tf_contrib.layers.xavier_initializer()
weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (self.filters,)


def batch_norm(x, name="batch_norm"):
    # return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)
    return tflow.keras.layers.BatchNormalization(name="BatchNorm", scale=True, center=True, trainable=True, epsilon=1e-5)(x)


def instance_norm(input, name="instance_norm", use_demod=False):
    if use_demod:
        return input

    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", use_demod=False):
    with tf.variable_scope(name):
        if use_demod:
            return conv2ddemod(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)
        else:
            return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d", use_upsampling=False, use_demod=False):
    # From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md 
    with tf.variable_scope(name):
        if use_upsampling:
            batch, in_height, in_width, in_channels = [int(d) for d in input_.get_shape()]
            x = tf.image.resize_nearest_neighbor(input_, (2*in_height,2*in_width))

            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            if use_demod:
                x = conv2ddemod(x, output_dim, ks, 1, padding="VALID", activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)
            else:
                x = slim.conv2d(x, output_dim, ks, 1, padding="VALID", activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)
            return x
        else:
            return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)
        

def dilated_conv2d(input_, output_dim, ks=3, s=2, stddev=0.02, padding='SAME', name="conv2d", use_demod=False):
    with tf.variable_scope(name):
        if use_demod:
            return conv2ddemod(input_, output_dim, ks, rate=s, padding=padding, activation_fn=None,
                               weights_initializer=tf.truncated_normal_initializer(0, stddev=stddev),
                               biases_initializer=None)
        else:
            batch, in_height, in_width, in_channels = [int(d) for d in input_.get_shape()]
            filter = tf.get_variable("filter", [ks, ks, in_channels, output_dim], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, stddev))
            conv = tf.nn.atrous_conv2d(input_,filter,rate=s,padding=padding,name=name)
            return conv



def one_step(x, ch, kernel, stride, name):
    return lrelu(instance_norm(conv2d(x, ch, kernel, stride, name=name + '_first_c'), name + '_first_bn'))

def one_step_dilated(x, ch, kernel, stride, name):
    return lrelu(instance_norm(dilated_conv2d(x, ch, kernel, stride, name=name + '_first_c'), name + '_first_bn'))

def num_steps(x, ch, kernel, stride,num_steps,name):
    for i in range(num_steps):
        x=lrelu(instance_norm(conv2d(x, ch, kernel, stride, name=name + '_c_'+str(i)), name + '_bn_'+str(i)))
    return x

def one_step_noins(x, ch, kernel, stride, name):
    return lrelu(conv2d(x, ch, kernel, stride, name=name + '_first_c'))

def num_steps_noins(x, ch, kernel, stride,num_steps,name):
    for i in range(num_steps):
        x=lrelu(conv2d(x, ch, kernel, stride, name=name + '_c_'+str(i)))
    return x


def dis_down(images,kernel_size,stride,n_scale,ch,name):
    backpack = images[0]
    for i in range(n_scale):
        if i == n_scale - 1:
            images[i] = num_steps(backpack,ch, kernel_size, stride, n_scale , name + str(i))
        else:
            images[i] = one_step_dilated(images[i + 1], ch, kernel_size, 1, name + str(i))
    return images


def dis_down_noins(images,kernel_size,stride,n_scale,ch,name):
    backpack = images[0]
    for i in range(n_scale):
        if i == n_scale - 1:
            images[i] = num_steps_noins(backpack,ch, kernel_size, stride, n_scale , name + str(i))
        else:
            images[i] = one_step_noins(images[i + 1], ch, kernel_size, 1, name + str(i))
    return images



def final_conv(images,n_scale,name):
    for i in range(n_scale):
        images[i]=conv2d(images[i], 1, s=1,name=name+str(i))
    return images


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def relu(x, leak=0.2, use_lrelu=False):
    if use_lrelu:
        return tf.nn.relu(x)
        name = 'lrelu'
    else:
        return tf.maximum(x, leak * x)
        name = 'relu'



def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def get_ones_like(logit):
    target=[]
    for i in range(len(logit)):
        target.append(tf.ones_like(logit[i]))
    return target

def get_zeros_like(logit):
    target=[]
    for i in range(len(logit)):
        target.append(tf.zeros_like(logit[i]))
    return target


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x



def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)

def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)