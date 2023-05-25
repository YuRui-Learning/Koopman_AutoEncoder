import torch.nn as nn
import torch
import math
import numpy as np
import tensorflow as tf

def define_loss( x, y, g_list):
    """Define the (unregularized) loss functions for the training.

        Arguments:
            x -- placeholder for input
            y -- list of outputs of network for each shift (each prediction step)
            g_list -- list of output of encoder for each shift (encoding each step in x)
            weights -- dictionary of weights for all networks
            biases -- dictionary of biases for all networks
            phase -- boolean placeholder for dropout: training phase or not training phase
            keep_prob -- probability that weight is kept during dropout

        Returns:
            loss1 -- autoencoder loss function
            loss2 -- dynamics/prediction loss function
            loss3 -- linearity loss function
            loss_Linf -- inf norm on autoencoder loss and one-step prediction loss
            loss -- sum of above four losses
    """
    # autoencoder loss: reconstruction loss
    y1 = []
    decoder_widths = [128, 64, 32, 16, 4]
    s_dim = 2
    state_weight = np.array([1, 1, 1])
    num_koopman_shifts = 1
    num_shifts = 68
    decoder_weights_num = len(decoder_widths) - 1
    denominator_nonzero = 10 ** (-5)
    loss1_denominator = tf.to_float(1.0)
    mean_squared_error = tf.reduce_mean(tf.reduce_mean(tf.square((y[0] - tf.squeeze(x[0, :, -s_dim:])) * 1), 1))
    loss1 = tf.truediv(mean_squared_error, loss1_denominator) # 转tensor或者float过程，truediv是一个除的表达式
    for i in np.arange(num_koopman_shifts - 1): # num_koopman_shifts - 1 不是0 ? 针对num_koopman_shifts不是0
        temp_y = decoder_apply(g_list[i + 1], self.weights, self.biases, args['dact_type'],args['batch_flag'], self.phase, 1, decoder_weights_num) # DECODER 求解
        loss1 = loss1 + tf.reduce_mean(tf.reduce_mean(tf.square((temp_y - tf.squeeze(x[i + 1, :, -s_dim:])) * state_weight), 1))
    loss1 = 5 * loss1 / num_koopman_shifts # recon_lam 正则化项 5 num_koopman_shifts是1

    # gets predicted loss
    loss2_denominator = tf.to_float(1.0)
    loss2 = tf.zeros([1, ], dtype=tf.float32)
    for i in np.arange(num_shifts):
        shift = i + 1 # 1
        # args['recon_lam'] 是5
        # x 和 y 求相减，这个不是应该是量化A B 的作用吗
        loss2 = loss2 + 5* tf.truediv(tf.reduce_mean(tf.reduce_mean(tf.square((y[shift] - tf.squeeze(x[shift, :, -s_dim:])) * state_weight), 1)),loss2_denominator)
    loss2 = loss2 / num_shifts

    # linear loss
    loss3 = tf.zeros([1, ], dtype=tf.float32)
    count_shift = 0
    y_lift = self.koopman_net(args, self.weights, self.biases, g_list[0], self.u[0, :, :], 1) # decoder结果
    for i in np.arange(num_koopman_shifts):
        loss3_denominator = tf.to_float(1.0)
        loss3 = loss3 + 5* tf.truediv(tf.reduce_mean(tf.reduce_mean(tf.square(y_lift - g_list[count_shift + 1]), 1)), loss3_denominator)
        count_shift += 1
        if (i + 1) < num_koopman_shifts:
            y_lift = self.koopman_net(args, self.weights, self.biases, y_lift, self.u[i + 1, :, :], 1) # 本身decoder回调进去的感觉
    loss3 = loss3 / num_koopman_shifts

    # 无穷范数 decoder 减去 groudn truth得到偏差，这里为什么是这样的？这里感觉是ground truth减去输入这个不是相当于loss1 求无穷范数吗？
    Linf1_den = tf.to_float(1.0)
    Linf2_den = tf.to_float(1.0)
    Linf1_penalty = tf.truediv(tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, -s_dim:]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
    Linf2_penalty = tf.truediv(tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, -s_dim:]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
    loss_Linf = 5 * (Linf1_penalty + Linf2_penalty)
    loss = loss1 + loss2 + loss3 + loss_Linf
    return loss1, loss2, loss3, loss_Linf, loss

def decoder_apply(self, prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, num_decoder_weights):
    """Apply a decoder to data prev_layer

    Arguments:
        prev_layer -- input to decoder network
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        num_decoder_weights -- number of weight matrices (layers) in decoder network

    Returns:
        output of decoder network applied to input prev_layer

    Side effects:
        None
    """
    num_decoder_weights = int(num_decoder_weights)
    for i in np.arange(num_decoder_weights):
        # if (i < num_decoder_weights - 1):
        h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
        # else:
        #     h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)])
        if batch_flag:
            h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
        if act_type[i] == 'sigmoid':
            h1 = tf.sigmoid(h1)
        elif act_type[i] == 'relu':
            h1 = tf.nn.relu(h1)
        elif act_type[i] == 'elu':
            h1 = tf.nn.elu(h1)
        elif act_type[i] == 'sin':
            h1 = tf.sin(h1)
        elif act_type[i] == 'cos':
            h1 = tf.cos(h1)
        elif act_type[i] == 'tanh':
            h1 = tf.tanh(h1)
        # the last layer don't need the dropout op
        if (i < num_decoder_weights - 1):
            prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)
        else:
            prev_layer = h1
    # apply last layer without any nonlinearity
    # last_layer = tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]
    return prev_layer