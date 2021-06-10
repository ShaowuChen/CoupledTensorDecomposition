

import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
from PIL import Image
import math

CS_ratio = 25  # 4, 10, 25, 30,, 40, 50
cpkt_model_number = 10

if CS_ratio == 4:
    n_input = 43
elif CS_ratio == 1:
    n_input = 10
elif CS_ratio == 10:
    n_input = 109
elif CS_ratio == 25:
    n_input = 272
elif CS_ratio == 30:
    n_input = 327
elif CS_ratio == 40:
    n_input = 436
elif CS_ratio == 50:
    n_input = 545

n_output = 1089
batch_size = 64
PhaseNumber = 5
nrtrain = 88912
learning_rate = 0.0001
EpochNum = 9
r1 = 64
r2 = 19

print('Load Data...')

Phi_data_Name = 'phi_0_%d_1089.mat' % CS_ratio
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi'].transpose()

Training_data_Name = 'Training_Data_Img91.mat'
Training_data = sio.loadmat(Training_data_Name)
Training_inputs = Training_data['inputs']
Training_labels = Training_data['labels']

# Computing Initialization Matrix
XX = Training_labels.transpose()
BB = np.dot(Phi_input.transpose(), XX)
BBB = np.dot(BB, BB.transpose())
CCC = np.dot(XX, BB.transpose())
PhiT_ = np.dot(CCC, np.linalg.inv(BBB))
del XX, BB, BBB, Training_data
PhiInv_input = PhiT_.transpose()
PhiTPhi_input = np.dot(Phi_input, Phi_input.transpose())

Test_Img = './Test_Image'


def imread_CS_py(imgName):
    block_size = 33
    Iorg = np.array(Image.open(imgName), dtype='float32')
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


filepaths = glob.glob(Test_Img + '/*.tif')

Phi = tf.constant(Phi_input, dtype=tf.float32)
PhiTPhi = tf.constant(PhiTPhi_input, dtype=tf.float32)
PhiInv = tf.constant(PhiInv_input, dtype=tf.float32)

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

X0 = tf.matmul(X_input, PhiInv)

PhiTb = tf.matmul(X_input, tf.transpose(Phi))


def norm(w, rank):
    dict = TT_compression(w, rank, rank)
    W = TT_compression_restore(dict)
    norm = np.linalg.norm(W)
    return norm


def TT_compression(weight, rank1, rank2):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]
    W = np.reshape(weight, [-1, O])
    u1, s1, v1 = np.linalg.svd(W, full_matrices=True)
    U1 = np.dot(u1[:, 0:rank1].copy(), np.diag(s1)[0:rank1, 0:rank1].copy())
    V1 = v1[0:rank1, :].copy()
    G12 = np.reshape(np.transpose(np.reshape(U1, [F1, F2, I, rank1]), (2, 0, 1, 3)), [I, -1])
    G3 = np.reshape(V1, [1, 1, rank1, O])
    u2, s2, v2 = np.linalg.svd(G12, full_matrices=True)
    U2 = np.dot(u2[:, 0:rank2].copy(), np.diag(s2)[0:rank2, 0:rank2].copy())
    V2 = v2[0:rank2, :].copy()
    G1 = np.reshape(U2, [1, 1, I, rank2])
    G2 = np.transpose(np.reshape(V2, [rank2, F1, F2, rank1]), (1, 2, 0, 3))
    return [G1, G2, G3]


def TT_compression_restore(dict):
    G1 = dict[0]
    G2 = dict[1]
    G3 = dict[2]
    m1, n1, j1, k1 = G1.shape
    m2, n2, j2, k2 = G2.shape
    m3, n3, j3, k3 = G3.shape
    G1 = np.reshape(G1, [m1 * n1 * j1, k1])
    G2 = np.reshape(np.transpose(G2, (2, 0, 1, 3)), [j2, -1])
    G12 = np.dot(G1, G2)

    G12 = np.reshape(np.transpose(np.reshape(G12, [j1, m2, n2, k2]), (1, 2, 0, 3)), [-1, k2])
    G3 = np.reshape(G3, [-1, k3])
    W = np.reshape(np.dot(G12, G3), [m2, n2, j1, k3])
    return W


def joint_compression(param_dict, aver_dict, rank1, rank2):
    # 分解不同部分
    # 第二层
    # 以后有时间可以写一个for循环
    w1_O_conv0, W1_1_conv0, W1_2_conv0 = TT_compression(
        param_dict['conv_0'][1] - TT_compression_restore(TT_compression(aver_dict[0], rank1, rank1)), rank2, rank2)
    w1_O_conv1, W1_1_conv1, W1_2_conv1 = TT_compression(
        param_dict['conv_1'][1] - TT_compression_restore(TT_compression(aver_dict[0], rank1, rank1)), rank2, rank2)
    w1_O_conv2, W1_1_conv2, W1_2_conv2 = TT_compression(
        param_dict['conv_2'][1] - TT_compression_restore(TT_compression(aver_dict[0], rank1, rank1)), rank2, rank2)
    w1_O_conv3, W1_1_conv3, W1_2_conv3 = TT_compression(
        param_dict['conv_3'][1] - TT_compression_restore(TT_compression(aver_dict[0], rank1, rank1)), rank2, rank2)
    w1_O_conv4, W1_1_conv4, W1_2_conv4 = TT_compression(
        param_dict['conv_4'][1] - TT_compression_restore(TT_compression(aver_dict[0], rank1, rank1)), rank2, rank2)


    # 第三层
    w11_O_conv0, W11_1_conv0, W11_2_conv0 = TT_compression(
        param_dict['conv_0'][2] - TT_compression_restore(TT_compression(aver_dict[1], rank1, rank1)), rank2, rank2)
    w11_O_conv1, W11_1_conv1, W11_2_conv1 = TT_compression(
        param_dict['conv_1'][2] - TT_compression_restore(TT_compression(aver_dict[1], rank1, rank1)), rank2, rank2)
    w11_O_conv2, W11_1_conv2, W11_2_conv2 = TT_compression(
        param_dict['conv_2'][2] - TT_compression_restore(TT_compression(aver_dict[1], rank1, rank1)), rank2, rank2)
    w11_O_conv3, W11_1_conv3, W11_2_conv3 = TT_compression(
        param_dict['conv_3'][2] - TT_compression_restore(TT_compression(aver_dict[1], rank1, rank1)), rank2, rank2)
    w11_O_conv4, W11_1_conv4, W11_2_conv4 = TT_compression(
        param_dict['conv_4'][2] - TT_compression_restore(TT_compression(aver_dict[1], rank1, rank1)), rank2, rank2)

    # 第四层
    w2_O_conv0, W2_1_conv0, W2_2_conv0 = TT_compression(
        param_dict['conv_0'][3] - TT_compression_restore(TT_compression(aver_dict[2], rank1, rank1)), rank2, rank2)
    w2_O_conv1, W2_1_conv1, W2_2_conv1 = TT_compression(
        param_dict['conv_1'][3] - TT_compression_restore(TT_compression(aver_dict[2], rank1, rank1)), rank2, rank2)
    w2_O_conv2, W2_1_conv2, W2_2_conv2 = TT_compression(
        param_dict['conv_2'][3] - TT_compression_restore(TT_compression(aver_dict[2], rank1, rank1)), rank2, rank2)
    w2_O_conv3, W2_1_conv3, W2_2_conv3 = TT_compression(
        param_dict['conv_3'][3] - TT_compression_restore(TT_compression(aver_dict[2], rank1, rank1)), rank2, rank2)
    w2_O_conv4, W2_1_conv4, W2_2_conv4 = TT_compression(
        param_dict['conv_4'][3] - TT_compression_restore(TT_compression(aver_dict[2], rank1, rank1)), rank2, rank2)

    # 第三层
    w22_O_conv0, W22_1_conv0, W22_2_conv0 = TT_compression(
        param_dict['conv_0'][4] - TT_compression_restore(TT_compression(aver_dict[3], rank1, rank1)), rank2, rank2)
    w22_O_conv1, W22_1_conv1, W22_2_conv1 = TT_compression(
        param_dict['conv_1'][4] - TT_compression_restore(TT_compression(aver_dict[3], rank1, rank1)), rank2, rank2)
    w22_O_conv2, W22_1_conv2, W22_2_conv2 = TT_compression(
        param_dict['conv_2'][4] - TT_compression_restore(TT_compression(aver_dict[3], rank1, rank1)), rank2, rank2)
    w22_O_conv3, W22_1_conv3, W22_2_conv3 = TT_compression(
        param_dict['conv_3'][4] - TT_compression_restore(TT_compression(aver_dict[3], rank1, rank1)), rank2, rank2)
    w22_O_conv4, W22_1_conv4, W22_2_conv4 = TT_compression(
        param_dict['conv_4'][4] - TT_compression_restore(TT_compression(aver_dict[3], rank1, rank1)), rank2, rank2)


    diff_dicts = {'conv_0': [w1_O_conv0, W1_1_conv0, W1_2_conv0, w11_O_conv0, W11_1_conv0, W11_2_conv0,
                             w2_O_conv0, W2_1_conv0, W2_2_conv0, w22_O_conv0, W22_1_conv0, W22_2_conv0],
                  'conv_1': [w1_O_conv1, W1_1_conv1, W1_2_conv1, w11_O_conv1, W11_1_conv1, W11_2_conv1,
                             w2_O_conv1, W2_1_conv1, W2_2_conv1, w22_O_conv1, W22_1_conv1, W22_2_conv1],
                  'conv_2': [w1_O_conv2, W1_1_conv2, W1_2_conv2, w11_O_conv2, W11_1_conv2, W11_2_conv2,
                             w2_O_conv2, W2_1_conv2, W2_2_conv2, w22_O_conv2, W22_1_conv2, W22_2_conv2],
                  'conv_3': [w1_O_conv3, W1_1_conv3, W1_2_conv3, w11_O_conv3, W11_1_conv3, W11_2_conv3,
                             w2_O_conv3, W2_1_conv3, W2_2_conv3, w22_O_conv3, W22_1_conv3, W22_2_conv3],
                  'conv_4': [w1_O_conv4, W1_1_conv4, W1_2_conv4, w11_O_conv4, W11_1_conv4, W11_2_conv4,
                             w2_O_conv4, W2_1_conv4, W2_2_conv4, w22_O_conv4, W22_1_conv4, W22_2_conv4],

                  }
    return diff_dicts


def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]


def ista_block(dicts, aver_weights, diff_dict, input_layers, input_data, layer_no):
    tau_value = tf.Variable(0.1, dtype=tf.float32)
    lambda_step = tf.Variable(0.1, dtype=tf.float32)
    soft_thr = tf.Variable(0.1, dtype=tf.float32)
    conv_size = 128
    filter_size = 3

    x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(input_layers[-1], PhiTPhi)),
                     tf.scalar_mul(lambda_step, PhiTb))  # X_k - lambda*A^TAX

    x2_ista = tf.reshape(x1_ista, shape=[-1, 33, 33, 1])

    Weights0 = tf.Variable(dicts[0], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights0')

    # 共享部分的参数分解
    Weights1_0_s = tf.get_variable(name='Weights1_0_s',
                                   initializer=tf.constant(TT_compression(aver_weights[0], r1, r1)[0]),
                                   dtype=tf.float32)
    Weights1_1_s = tf.get_variable(name='Weights1_1_s',
                                   initializer=tf.constant(TT_compression(aver_weights[0], r1, r1)[1]),
                                   dtype=tf.float32)
    Weights1_2_s = tf.get_variable(name='Weights1_2_s',
                                   initializer=tf.constant(TT_compression(aver_weights[0], r1, r1)[2]),
                                   dtype=tf.float32)
    Weights11_0_s = tf.get_variable(name='Weights11_0_s',
                                    initializer=tf.constant(TT_compression(aver_weights[1], r1, r1)[0]),
                                    dtype=tf.float32)
    Weights11_1_s = tf.get_variable(name='Weights11_1_s',
                                    initializer=tf.constant(TT_compression(aver_weights[1], r1, r1)[1]),
                                    dtype=tf.float32)
    Weights11_2_s = tf.get_variable(name='Weights11_2_s',
                                    initializer=tf.constant(TT_compression(aver_weights[1], r1, r1)[2]),
                                    dtype=tf.float32)
    Weights2_0_s = tf.get_variable(name='Weights2_0_s',
                                   initializer=tf.constant(TT_compression(aver_weights[2], r1, r1)[0]),
                                   dtype=tf.float32)
    Weights2_1_s = tf.get_variable(name='Weights2_1_s',
                                   initializer=tf.constant(TT_compression(aver_weights[2], r1, r1)[1]),
                                   dtype=tf.float32)
    Weights2_2_s = tf.get_variable(name='Weights2_2_s',
                                   initializer=tf.constant(TT_compression(aver_weights[2], r1, r1)[2]),
                                   dtype=tf.float32)
    Weights22_0_s = tf.get_variable(name='Weights22_0_s',
                                    initializer=tf.constant(TT_compression(aver_weights[3], r1, r1)[0]),
                                    dtype=tf.float32)
    Weights22_1_s = tf.get_variable(name='Weights22_1_s',
                                    initializer=tf.constant(TT_compression(aver_weights[3], r1, r1)[1]),
                                    dtype=tf.float32)
    Weights22_2_s = tf.get_variable(name='Weights22_2_s',
                                    initializer=tf.constant(TT_compression(aver_weights[3], r1, r1)[2]),
                                    dtype=tf.float32)

    # 不同部分的参数分解
    Weights1_0_d = tf.Variable(diff_dict[0], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights1_0_d')
    Weights1_1_d = tf.Variable(diff_dict[1], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights1_1_d')
    Weights1_2_d = tf.Variable(diff_dict[2], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights1_2_d')
    Weights11_0_d = tf.Variable(diff_dict[3], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights11_0_d')
    Weights11_1_d = tf.Variable(diff_dict[4], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights11_1_d')
    Weights11_2_d = tf.Variable(diff_dict[5], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights11_2_d')
    Weights2_0_d = tf.Variable(diff_dict[6], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights2_0_d')
    Weights2_1_d = tf.Variable(diff_dict[7], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights2_1_d')
    Weights2_2_d = tf.Variable(diff_dict[8], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights2_2_d')
    Weights22_0_d = tf.Variable(diff_dict[9], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights22_0_d')
    Weights22_1_d = tf.Variable(diff_dict[10], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights22_1_d')
    Weights22_2_d = tf.Variable(diff_dict[11], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights22_2_d')

    Weights3 = tf.Variable(dicts[5], dtype=tf.float32, name='conv_' + str(layer_no) + '/' + 'Weights3')
    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    # TT分解之后的网络共享的W->G1,G2,G3，不同的Wn->G1n,G2n,G3n
    # 第二层共享参数分解
    x4_0_ista_s = tf.nn.conv2d(x3_ista, Weights1_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x4_1_ista_s = tf.nn.conv2d(x4_0_ista_s, Weights1_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x4_2_ista_s = tf.nn.conv2d(x4_1_ista_s, Weights1_2_s, strides=[1, 1, 1, 1], padding='SAME')
    # 第二层不同参数分解
    x4_0_ista_d = tf.nn.conv2d(x3_ista, Weights1_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_1_ista_d = tf.nn.conv2d(x4_0_ista_d, Weights1_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_2_ista_d = tf.nn.conv2d(x4_1_ista_d, Weights1_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_ista = tf.nn.relu(tf.add(x4_2_ista_s, x4_2_ista_d))
    # 第三层共享参数分解
    x44_0_ista_s = tf.nn.conv2d(x4_ista, Weights11_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x44_1_ista_s = tf.nn.conv2d(x44_0_ista_s, Weights11_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x44_2_ista_s = tf.nn.conv2d(x44_1_ista_s, Weights11_2_s, strides=[1, 1, 1, 1], padding='SAME')
    # 第三层不同参数分解
    x44_0_ista_d = tf.nn.conv2d(x4_ista, Weights11_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x44_1_ista_d = tf.nn.conv2d(x44_0_ista_d, Weights11_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x44_2_ista_d = tf.nn.conv2d(x44_1_ista_d, Weights11_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x44_ista = tf.add(x44_2_ista_s, x44_2_ista_d)
    x5_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - soft_thr))
    # 第四层共享参数分解
    x6_0_ista_s = tf.nn.conv2d(x5_ista, Weights2_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x6_1_ista_s = tf.nn.conv2d(x6_0_ista_s, Weights2_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x6_2_ista_s = tf.nn.conv2d(x6_1_ista_s, Weights2_2_s, strides=[1, 1, 1, 1], padding='SAME')
    # 第四层不同参数分解
    x6_0_ista_d = tf.nn.conv2d(x5_ista, Weights2_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_1_ista_d = tf.nn.conv2d(x6_0_ista_d, Weights2_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_2_ista_d = tf.nn.conv2d(x6_1_ista_d, Weights2_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_ista = tf.nn.relu(tf.add(x6_2_ista_s, x6_2_ista_d))
    # 第五层共享参数分解
    x66_0_ista_s = tf.nn.conv2d(x6_ista, Weights22_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x66_1_ista_s = tf.nn.conv2d(x66_0_ista_s, Weights22_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x66_2_ista_s = tf.nn.conv2d(x66_1_ista_s, Weights22_2_s, strides=[1, 1, 1, 1], padding='SAME')
    # 第五层不同参数分解
    x66_0_ista_d = tf.nn.conv2d(x6_ista, Weights22_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x66_1_ista_d = tf.nn.conv2d(x66_0_ista_d, Weights22_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x66_2_ista_d = tf.nn.conv2d(x66_1_ista_d, Weights22_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x66_ista = tf.add(x66_2_ista_s, x66_2_ista_d)
    x7_ista = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x7_ista + x2_ista

    x8_ista = tf.reshape(x7_ista, shape=[-1, 1089])

    x3_0_ista_sym_s = tf.nn.conv2d(x3_ista, Weights1_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x3_1_ista_sym_s = tf.nn.conv2d(x3_0_ista_sym_s, Weights1_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x3_2_ista_sym_s = tf.nn.conv2d(x3_1_ista_sym_s, Weights1_2_s, strides=[1, 1, 1, 1], padding='SAME')
    x3_0_ista_sym_d = tf.nn.conv2d(x3_ista, Weights1_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x3_1_ista_sym_d = tf.nn.conv2d(x3_0_ista_sym_d, Weights1_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x3_2_ista_sym_d = tf.nn.conv2d(x3_1_ista_sym_d, Weights1_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x3_ista_sym = tf.nn.relu(tf.add(x3_2_ista_sym_s, x3_2_ista_sym_d))

    x4_0_ista_sym_s = tf.nn.conv2d(x3_ista_sym, Weights11_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x4_1_ista_sym_s = tf.nn.conv2d(x4_0_ista_sym_s, Weights11_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x4_2_ista_sym_s = tf.nn.conv2d(x4_1_ista_sym_s, Weights11_2_s, strides=[1, 1, 1, 1], padding='SAME')
    x4_0_ista_sym_d = tf.nn.conv2d(x3_ista_sym, Weights11_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_1_ista_sym_d = tf.nn.conv2d(x4_0_ista_sym_d, Weights11_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_2_ista_sym_d = tf.nn.conv2d(x4_1_ista_sym_d, Weights11_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x4_ista_sym = tf.add(x4_2_ista_sym_s, x4_2_ista_sym_d)

    x6_0_ista_sym_s = tf.nn.conv2d(x4_ista_sym, Weights2_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x6_1_ista_sym_s = tf.nn.conv2d(x6_0_ista_sym_s, Weights2_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x6_2_ista_sym_s = tf.nn.conv2d(x6_1_ista_sym_s, Weights2_2_s, strides=[1, 1, 1, 1], padding='SAME')
    x6_0_ista_sym_d = tf.nn.conv2d(x4_ista_sym, Weights2_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_1_ista_sym_d = tf.nn.conv2d(x6_0_ista_sym_d, Weights2_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_2_ista_sym_d = tf.nn.conv2d(x6_1_ista_sym_d, Weights2_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x6_ista_sym = tf.nn.relu(tf.add(x6_2_ista_sym_s, x6_2_ista_sym_d))

    x7_0_ista_sym_s = tf.nn.conv2d(x6_ista_sym, Weights22_0_s, strides=[1, 1, 1, 1], padding='SAME')
    x7_1_ista_sym_s = tf.nn.conv2d(x7_0_ista_sym_s, Weights22_1_s, strides=[1, 1, 1, 1], padding='SAME')
    x7_ista_sym_s = tf.nn.conv2d(x7_1_ista_sym_s, Weights22_2_s, strides=[1, 1, 1, 1], padding='SAME')
    x7_0_ista_sym_d = tf.nn.conv2d(x6_ista_sym, Weights22_0_d, strides=[1, 1, 1, 1], padding='SAME')
    x7_1_ista_sym_d = tf.nn.conv2d(x7_0_ista_sym_d, Weights22_1_d, strides=[1, 1, 1, 1], padding='SAME')
    x7_ista_sym_d = tf.nn.conv2d(x7_1_ista_sym_d, Weights22_2_d, strides=[1, 1, 1, 1], padding='SAME')
    x7_ista_sym = tf.add(x7_ista_sym_s, x7_ista_sym_d)

    x11_ista = x7_ista_sym - x3_ista
    var_list_s = [Weights1_0_s, Weights1_1_s, Weights1_2_s, Weights11_0_s, Weights11_1_s, Weights11_2_s,
                  Weights2_0_s, Weights2_1_s, Weights2_2_s, Weights22_0_s, Weights22_1_s, Weights22_2_s]
    var_list_d = [Weights1_0_d, Weights1_1_d, Weights1_2_d, Weights11_0_d, Weights11_1_d, Weights11_2_d,
                  Weights2_0_d, Weights2_1_d, Weights2_2_d, Weights22_0_d, Weights22_1_d, Weights22_2_d
                  ]
    var_list = [Weights1_0_s, Weights1_1_s, Weights1_2_s, Weights11_0_s, Weights11_1_s, Weights11_2_s,
                Weights2_0_s, Weights2_1_s, Weights2_2_s, Weights22_0_s, Weights22_1_s, Weights22_2_s,
                Weights1_0_d, Weights1_1_d, Weights1_2_d, Weights11_0_d, Weights11_1_d, Weights11_2_d,
                Weights2_0_d, Weights2_1_d, Weights2_2_d, Weights22_0_d, Weights22_1_d, Weights22_2_d
                ]
    return [x8_ista, x11_ista], var_list_s, var_list_d, var_list


def inference_ista(param_dict, aver_weight, diff_comp, input_tensor, n, X_output):
    layers = []
    layers_symetric = []
    var_d = []
    loss_var = []
    layers.append(input_tensor)
    for i in range(n):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            [conv1, conv1_sym], var_s, var_diff, var_list = ista_block(param_dict['conv_%d' % i], aver_weight,
                                                                       diff_comp['conv_%d' % i], layers, X_output, i)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
            var_d.append(var_diff)
            loss_var.append(var_list)

    return [layers, layers_symetric], var_s, var_d, loss_var


param_dict = np.load('./params_dict.npy', encoding='latin1', allow_pickle=True).item()
aver_weights = np.load('./5_25_aver_after_finetune.npy', encoding='latin1', allow_pickle=True).item()

diff_compression = joint_compression(param_dict, aver_weights['aver_weights'], r1, r2)

[Prediction, Pre_symetric], Var_s, Var_d, Loss_Var = inference_ista(param_dict, aver_weights['aver_weights'],
                                                                    diff_compression, X0, PhaseNumber, X_output)

cost0 = tf.reduce_mean(tf.square(X0 - X_output))


def compute_share_weights_norm_rate():
    norm_tt = 0
    norm_aver = 0
    for i in range(4):
        # print(Var_s[3*i:3*i+3][0].shape)
        # print(Var_s[3 * i:3 * i + 3][0])
        G1 = Var_s[3 * i]
        G2 = Var_s[3 * i + 1]
        G3 = Var_s[3 * i + 2]
        w = [G1, G2, G3]
        W_tt = TT_compression_restore(w)
        norm1 = np.linalg.norm(W_tt)
        norm_tt += norm1
        norm2 = np.linalg.norm(aver_weights['aver_weights'][i])
        norm_aver += norm2
    norm_rate_s = norm_tt / norm_aver
    return norm_rate_s


def compute_different_weights_norm_rate():
    norm_tt = 0
    norm_diff = 0
    for i in range(5):
        for j in range(4):
            W_tt = TT_compression_restore(Var_d[i][3 * j:3 * j + 3])
            norm1 = np.linalg.norm(W_tt)
            norm_tt += norm1
    for m in range(5):
        for n in range(4):
            W_diff = param_dict['conv_' + str(m)][n + 1] - TT_compression_restore(Var_s[3 * n:3 * n + 3])
            norm2 = np.linalg.norm(W_diff)
            norm_diff += norm2
    norm_rate_d = norm_tt / norm_diff
    return norm_rate_d


def compute_cost(Prediction, X_output, PhaseNumber):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))

    return [cost, cost_sym]


[cost, cost_sym] = compute_cost(Prediction, X_output, PhaseNumber)

cost_all = cost + 0.01 * cost_sym

optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all, var_list=Loss_Var)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, CS_ratio))
print("...............................\n")

print("Strart Training..")

model_dir = 'Phase_%d_ratio_0_%d_ISTA_Net_plus_TT_r1%d_r2%d_compression_Model' % (PhaseNumber, CS_ratio, r1, r2)

output_file_name1 = "Log_output_%s.txt" % (model_dir)

for epoch_i in range(0, EpochNum + 1):
    randidx_all = np.random.permutation(nrtrain)
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i * batch_size:(batch_i + 1) * batch_size]

        batch_ys = Training_labels[randidx, :]
        batch_xs = np.dot(batch_ys, Phi_input)

        feed_dict = {X_input: batch_xs, X_output: batch_ys}
        sess.run(optm_all, feed_dict=feed_dict)

    output_data = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (
    epoch_i, EpochNum, sess.run(cost, feed_dict=feed_dict), sess.run(cost_sym, feed_dict=feed_dict))
    print(output_data)

    output_file1 = open(output_file_name1, 'a')
    output_file1.write(output_data)
    output_file1.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=True)
# modelP_path = os.path.join('.', '4_weights_TT_compression_after_finetune.npy')
# np.save(modelP_path, Loss_Var)
print("Training Finished")

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
output_file_name2 = "PSNR_Results_%s.txt" % (model_dir)
output_file2 = open(output_file_name2, 'a')

for img_no in range(ImgNum):
    imgName = filepaths[img_no]

    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
    Icol = img2col_py(Ipad, 33).transpose() / 255.0
    print(Ipad.shape)
    Img_input = np.dot(Icol, Phi_input)
    Img_output = Icol

    start = time()
    Prediction_value = sess.run(Prediction[-1], feed_dict={X_input: Img_input})

    end = time()
    cost_sym_value = sess.run(cost_sym, feed_dict={X_input: Img_input, X_output: Img_output})

    X_rec = col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new)

    rec_PSNR = psnr(X_rec * 255, Iorg)

    print(
        "Run time for %s is %.4f, PSNR is %.2f, loss sym is %.4f" % (imgName, (end - start), rec_PSNR, cost_sym_value))

    img_rec_name = "%s_rec_%s_%d_PSNR_%.2f.png" % (imgName, model_dir, cpkt_model_number, rec_PSNR)

    x_im_rec = Image.fromarray(np.clip(X_rec * 255, 0, 255).astype(np.uint8))
    x_im_rec.save(img_rec_name)

    PSNR_All[0, img_no] = rec_PSNR

output_data = "Avg PSNR is %.2f dB, cpkt NO. is %d \n" % (np.mean(PSNR_All), cpkt_model_number)
# Norm_rate_s=compute_share_weights_norm_rate()
# Norm_rate_d=compute_different_weights_norm_rate()
print(output_data)
# print("Norm_rate_s = "+str(Norm_rate_s)+"\nNorm_rate_d = "+str(Norm_rate_d))
output_file2.write(output_data)
output_file2.close()
print("Reconstruction READY")
sess.close()
