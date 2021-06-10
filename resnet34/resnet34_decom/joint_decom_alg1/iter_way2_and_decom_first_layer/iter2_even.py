import tensorflow as tf
import numpy as np

import os
from tensorflow.python import pywrap_tensorflow


flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('rank_rate_yellow', None, 'rank_rate of each block1 conv2')
flags.DEFINE_string('gpu', '0', 'rank_rate of each block1 conv2')
flags.DEFINE_string('ckpt_path', '20200727', 'rank_rate of each block1 conv2')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")
flags.DEFINE_string('whole_ave', 'False', 'deecompose W_ave or not')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_yellow'+FLAGS.rank_rate_yellow[0]+'_'+FLAGS.rank_rate_yellow[2]+'_rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]
ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]

os.makedirs(ckpt_path,exist_ok=True)



def decompose(value, rank_rate):
    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    rank1=rank2=int(np.floor(i*rank_rate))

    # [H,W,I,O]
    value_2d = np.reshape(value, [h * w * i, o])
    U2, S2, V2 = np.linalg.svd(value_2d)


    # max rank: o
    V2 = np.matmul(np.diag(S2)[:rank2, :rank2], V2[:rank2, :])
    U2_cut = U2[:, :rank2]

    # to [i,h,w,rank2] and then [i, hw*rank2]
    U2_cut = np.transpose(np.reshape(U2_cut, [h, w, i, rank2]), [2, 0, 1, 3])
    U2_cut = np.reshape(U2_cut, [i, h * w * rank2])

    U1, S1, V1 = np.linalg.svd(U2_cut)
    # max rank: i
    assert(rank1<=len(S1))
    V1 = np.matmul(np.diag(S1)[:rank1, :rank1], V1[:rank1, :])
    U1_cut = U1[:, :rank1]

    G3 = np.reshape(V2, [1, 1, rank2, o])
    G2 = np.transpose(np.reshape(V1, [rank1, h, w, rank2]), [1, 2, 0, 3])

    G1 = np.reshape(U1_cut, [1, 1, i, rank1])


    return G3,G2,G1

def reverse_decompose(G_ave3,G_ave2,G_ave1):
    G_ave3_shape = G_ave3.shape
    G_ave2_shape = G_ave2.shape
    G_ave1_shape = G_ave1.shape

    I = G_ave1_shape[2]
    O = G_ave3_shape[3]
    H = 3
    W = 3



    U1 = np.reshape(G_ave1, [G_ave1_shape[2], G_ave1_shape[3]])

    V1 = np.reshape( np.transpose(G_ave2,[2,0,1,3]), [G_ave2_shape[3],-1])


    U2_temp = np.matmul(U1,V1)
    U2 = np.reshape(np.transpose( np.reshape(U2_temp,[I,H,W,-1]), [1,2,0,3]), [H*W*I, -1])

    V2 = np.reshape(G_ave3, [-1,O])

    weight_reverse_2d = np.matmul(U2,V2)

    assert(weight_reverse_2d.shape==(H*W*I,O))
    weight_reverse_4d = np.reshape(weight_reverse_2d, [H,W,I,O])
    return weight_reverse_4d


def get_dict():
    orig_checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'

    orig_dict = {}
    orig_reader = pywrap_tensorflow.NewCheckpointReader(orig_checkpoint_path)
    orig_var_to_shape_map = orig_reader.get_variable_to_shape_map()

    for key in orig_var_to_shape_map:
        orig_dict[key] = orig_reader.get_tensor(key)


    return orig_dict


def iter(value_dict,rank_rate_ave,rank_rate_single):

    layer_nums = [2,3,4]
    repeats= [4,6,3]

    for i in range(3):  # layer
        layer_num = i + 2
        # for j in range(repeats[i] - 1):  # repeat
        #     repeat_num = j + 1

        '''W2， 即黃框第二層也加入分解行列中來'''
        for j in range(repeats[i]):  # repeat
            repeat_num = j

            for conv_num in [2]:
            # for conv_num in [1, 2]:  # conv
                orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'

                diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_diff'

                locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose(value_dict[orig_name], rank_rate_single)

                # locals()['layer' + str(layer_num) + '.conv' + str(conv_num) + '.apdate_W_diff'] = 0


    #迭代
    for iters in range(11):
        print('==============================================================================')
        print('ites',iters)

        value_dict_for_iter = value_dict.copy()

        collection_to_update_W_ave = {}
        #创建变量
        for i in range(3):#layer
            layer_num = i+2
            # for conv_num in [1,2]:
            for conv_num in [2]:
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'
                locals()[ave_name] = 0

        for i in range(3):  #layer
            layer_num = i+2
            # for j in range(repeats[i] - 1):  #repeat
            #     repeat_num = j + 1
            for j in range(repeats[i]):  #repeat
                repeat_num = j
                # for conv_num in [1, 2]:   #conv
                for conv_num in [2]:
                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_diff'

                    ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'
                    locals()[ave_name] += (value_dict[orig_name] - reverse_decompose( locals()[diff_name + '3'],   locals()[diff_name + '2'],  locals()[diff_name + '1']))/(repeats[i]-1)

                    for G in [1, 2, 3]:
                        value_dict_for_iter[diff_name + str(G)] = locals()[diff_name + str(G)]

        for i in range(3):#layer
            layer_num = i+2
            # for conv_num in [1,2]:
            for conv_num in [2]:
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'
                locals()[ave_name + '3'], locals()[ave_name + '2'], locals()[ave_name + '1'] = decompose(locals()[ave_name], rank_rate_ave)
                for G in [1, 2, 3]:
                    value_dict_for_iter[ave_name + str(G)] = locals()[ave_name + str(G)]

        if iters==0 or iters==3 or iters==5 or iters==10:
            np.save(ckpt_path+'/iter_even'+str(iters)+'.npy', value_dict_for_iter)
        if iters==1:
            for key in value_dict_for_iter:
                print(key)




        #更新W_diff和G_diff
        for i in range(3):  # layer
            layer_num = i + 2
            # for j in range(repeats[i] - 1):  # repeat
            #     repeat_num = j + 1
            for j in range(repeats[i]):  # repeat
                repeat_num = j
                # for conv_num in [1, 2]:  # conv
                for conv_num in [2]:
                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_diff'
                    ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

                    locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose(value_dict[orig_name]-reverse_decompose( locals()[ave_name + '3'],   locals()[ave_name + '2'],  locals()[ave_name + '1']), rank_rate_single)



rank_rate_ave = eval(FLAGS.rank_rate_ave)
rank_rate_single = eval(FLAGS.rank_rate_single)
value_dict = get_dict()
iter(value_dict,rank_rate_ave,rank_rate_single)