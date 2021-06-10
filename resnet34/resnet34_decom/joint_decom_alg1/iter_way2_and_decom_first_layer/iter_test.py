import tensorflow as tf
import numpy as np

import os
from tensorflow.python import pywrap_tensorflow


flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('rank_rate_yellow', None, 'rank_rate of each block1 conv2')
flags.DEFINE_string('gpu', '0', 'rank_rate of each block1 conv2')
flags.DEFINE_string('ckpt_path', '20200623', 'rank_rate of each block1 conv2')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")
flags.DEFINE_string('whole_ave', 'False', 'deecompose W_ave or not')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_yellow'+FLAGS.rank_rate_yellow[0]+'_'+FLAGS.rank_rate_yellow[2]+'_rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]
os.makedirs(ckpt_path,exist_ok=True)


''' 分開處理奇數層和單數層；然後在總程序中load兩個npy文件再合併即可'''



def get_dict():
    orig_checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'

    orig_dict = {}
    orig_reader = pywrap_tensorflow.NewCheckpointReader(orig_checkpoint_path)
    orig_var_to_shape_map = orig_reader.get_variable_to_shape_map()

    for key in orig_var_to_shape_map:
        orig_dict[key] = orig_reader.get_tensor(key)

    return orig_dict

def decompose_even(value, rank_rate):
    '''应用于偶数层分解'''
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

def reverse_decompose_even(G_ave3,G_ave2,G_ave1):
    '''应用于偶数层分解'''
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


def decompose_odd(value, rank_rate):
    '''应用于奇数层onv层，L/2'''

    #注意，此法将S1给了G23,而不给G1(可能有差别，也可能无差别；主要是觉得不同G1值小一点，差别可能也小一点；后面可以实验验证）

    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    rank2 = int(np.floor(i*rank_rate))
    rank1 = min(rank2, i)

    # [H,W,I,O] -->  [I, H, W, O]  --> [I, HWO]
    value_t = np.transpose(value, [2,0,1,3])
    value_2d = np.reshape(value_t, [i, h*w*o])


    G1_2d,S1, G23_2d = np.linalg.svd(value_2d)
    G1_2d = G1_2d[:,:rank1]  #[i, rank1]
    G23_2d = np.matmul(np.diag(S1)[:rank1, :rank1], G23_2d[:rank1,:])  #[rank1, hwo]

    assert (G1_2d.shape == [i, rank1])
    G1 = np.reshape(G1_2d, [1,1,i,rank1])

    G23_2d = np.reshape( np.rehshape(G23_2d, [rank1,h,w,o]), [rank1*h*w,o])
    G2_2d, S2, G3_2d = np.linalg.svd(G23_2d)
    G2_2d = G2_2d[:,:rank2]
    G3_2d = np.matmul(S2[:rank2, :rank2], G2_2d[:rank2,:])

    assert(G2_2d.shape == [rank1*h*w, rank2])
    G2 = np.transpose( np.reshape(G2_2d, [rank1, h, w, rank2]), [1,2,0,3])

    assert(G3_2d.shape == [rank2, o])
    G3 = np.reshape(G3_2d, [1,1,rank2,o])

    return G3,G2,G1

def reverse_decompose_odd(G3, G2, G1):
    '''应用于奇数层onv层，L/2'''

    i = G1.shape[2]
    o = G3.shape[3]
    r1 = G2.shape[2]
    r2 = G2.shape[3]

    G3_2d = np.reshape(G3, [r2, o])
    G2_2d = np.reshape(np.transpose(G2, [2, 0, 1, 3]), [ r1*3*3, r2])
    G23_2d = np.reshape( np.reshape( np.matmul(G2_2d, G3_2d), [r1, 3,3, o] ), [r1, 9*o])

    G1_2d = np.reshape( G1, [i, r1])

    weight_2d = np.matmul(G1_2d, G23_2d)
    weight = np.transpose( np.reshape(weight_2d, [i, 3, 3, o ]), [1,2,0,3]

    return weight



def iter_odd():
    return


def iter_even():
    layer_nums = [2, 3, 4]
    repeats = [4, 6, 3]


    return


def init_odd():
    return


def init_even(value_dict, rank_rate_single):
    repeats = [4, 6, 3]

    for i in range(3):  # layer
        layer_num = i + 2
        for repeat_num in range(repeats[i] - 1):  # repeat
            for conv_num in [1, 2]:  # conv
                w_num = conv_num + 2 * repeat_num

                if w_num % 2 == 0:
                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'

                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_odd_diff'

                    locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_even(value_dict[orig_name], rank_rate_single)


    return locals()


def iter():

    layer_nums = [2,3,4]
    repeats= [4,6,3]

    #初始化
    locals().update(init_even(value_dict, rank_rate_single))

   #迭代
    for iters in range(11):
        print('==============================================================================')
        print('ites',iters)

        value_dict_for_iter = value_dict.copy()

        #创建变量
        for i in range(3):#layer
            layer_num = i+2
            for conv_num in [1,2]:
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse_odd'
                locals()[ave_name] = 0








def iter(value_dict,rank_rate_ave,rank_rate_single):

    layer_nums = [2,3,4]
    repeats= [4,6,3]

    for i in range(3):  # layer
        layer_num = i + 2
        for j in range(repeats[i] - 1):  # repeat
            repeat_num = j
            for conv_num in [1, 2]:  # conv

                w_num = conv_num + 2 * (repeat_num - 1)
                if w_num %2==1:

                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'

                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_odd_diff'

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
            for conv_num in [1,2]:
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse_odd'
                locals()[ave_name] = 0

        for i in range(3):  #layer
            layer_num = i+2
            for j in range(repeats[i] - 1):  #repeat
                repeat_num = j + 1
                for conv_num in [1, 2]:   #conv
                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_odd'
                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_odd_diff'

                    ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse_odd'
                    locals()[ave_name] += (value_dict[orig_name] - reverse_decompose( locals()[diff_name + '3'],   locals()[diff_name + '2'],  locals()[diff_name + '1']))/(repeats[i]-1)

                    for G in [1, 2, 3]:
                        value_dict_for_iter[diff_name + str(G)] = locals()[diff_name + str(G)]

        for i in range(3):#layer
            layer_num = i+2
            for conv_num in [1,2]:
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse_odd'
                locals()[ave_name + '3'], locals()[ave_name + '2'], locals()[ave_name + '1'] = decompose(locals()[ave_name], rank_rate_ave)
                for G in [1, 2, 3]:
                    value_dict_for_iter[ave_name + str(G)] = locals()[ave_name + str(G)]

        if iters==0 or iters==3 or iters==5 or iters==10:
            np.save(ckpt_path+'/iter'+str(iters)+'.npy', value_dict_for_iter)





        #更新W_diff和G_diff
        for i in range(3):  # layer
            layer_num = i + 2
            for j in range(repeats[i] - 1):  # repeat
                repeat_num = j + 1
                for conv_num in [1, 2]:  # conv
                    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                    diff_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_diff'
                    ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

                    locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose(value_dict[orig_name]-reverse_decompose( locals()[ave_name + '3'],   locals()[ave_name + '2'],  locals()[ave_name + '1']), rank_rate_single)



rank_rate_ave = eval(FLAGS.rank_rate_ave)
rank_rate_single = eval(FLAGS.rank_rate_single)
value_dict = get_dict()
iter(value_dict,rank_rate_ave,rank_rate_single)