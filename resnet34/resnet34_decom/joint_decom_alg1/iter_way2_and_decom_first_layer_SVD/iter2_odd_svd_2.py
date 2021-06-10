import tensorflow as tf
import numpy as np

import os
from tensorflow.python import pywrap_tensorflow





def decompose_single(value, rank_rate):
    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    if i==o:
        rank2 = int(np.floor(o*rank_rate))
        rank1 = min(rank2, i)
        assert(rank1==rank2)

    else:
        rank2 = int(np.floor(o*rank_rate))
        rank1 = min(rank2, i)


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

def reverse_decompose_single(G_ave3,G_ave2,G_ave1):
    G_ave3_shape = G_ave3.shape
    G_ave2_shape = G_ave2.shape
    G_ave1_shape = G_ave1.shape

    I = G_ave1_shape[2]
    O = G_ave3_shape[3]
    H = 3
    W = 3

    U1 = np.reshape(G_ave1, [G_ave1_shape[2], G_ave1_shape[3]])

    V1 = np.reshape( np.transpose(G_ave2,[2,0,1,3]), [G_ave2_shape[2],-1])


    U2_temp = np.matmul(U1,V1)
    U2 = np.reshape(np.transpose( np.reshape(U2_temp,[I,H,W,-1]), [1,2,0,3]), [H*W*I, -1])

    V2 = np.reshape(G_ave3, [-1,O])

    weight_reverse_2d = np.matmul(U2,V2)

    assert(weight_reverse_2d.shape==(H*W*I,O))
    weight_reverse_4d = np.reshape(weight_reverse_2d, [H,W,I,O])
    return weight_reverse_4d
 

def decompose_W1(value, rank_rate):
    # print(value.shape)
    shape = value.shape
    i = shape[0]
    o = shape[1]

    o1 = o//9
    # print('o',o)
    # print('o1',o1)
    assert(o1*9==o)

    i1 = i //3
    i2 = i //3 *2
    assert(i1+i2==i)


    rank2 = int(np.floor(o1*rank_rate))

    #因为rank_ave<=1/2，所以以i2为基准，不会超标
    rank1 = int(np.floor(i2*rank_rate))

    #[I, 9o]
    G1,S,G23 = np.linalg.svd(value)
 
    G1 = G1[:, :rank1]
    G1_first = G1[:i1, :]
    G1_ave = G1[i1:, :]
    G1_first = np.reshape(G1_first, [1,1, i1, rank1])
    G1_ave = np.reshape(G1_ave, [1,1, i2, rank1])

    #[rank1, 9o] --> [rank1*9,o]
    G23 = np.matmul(np.diag(S)[:rank1,:rank1], G23[:rank1, :])
    G23 = np.reshape(np.reshape(G23, [rank1,9,o1]),[rank1*9,o1])
    G2,S,G3 = np.linalg.svd(G23)

    #[rank1*9,rank2]-->[3,3,rank1,rank2]
    G2 = G2[:,:rank2]
    G2 = np.transpose(np.reshape(np.reshape(G2,[rank1,9,rank2]), [rank1,3,3,rank2]), [1,2,0,3])

    #[rank2,o]--> [1,1,rank2,o]
    G3 = np.matmul(np.diag(S)[:rank2,:rank2], G3[:rank2,:])
    G3 = np.reshape(G3, [1,1,rank2,o1])

    return G3, G2, G1_ave, G1_first



def reverse_decompose_ave(G3, G2, G1):
    G3_shape = G3.shape
    G2_shape = G2.shape
    G1_shape = G1.shape

    G3 = np.reshape(G3, [G3_shape[2], G3_shape[3]])
    G2 = np.reshape(np.reshape( np.transpose(G2, [2,0,1,3]), [G2_shape[2], 9, G2_shape[3]]), [G2_shape[2]*9, G2_shape[3]])
    G23 = np.matmul(G2, G3)
    G23 = np.reshape(G23, [G2_shape[2],9,G3_shape[3]])
    G23 = np.reshape(G23,[G2_shape[2],-1])

    G1 = np.reshape(G1, [G1_shape[2], G1_shape[3]])
    
    value = np.matmul(G1, G23)
    value = np.transpose(np.reshape(np.reshape(value, [G1_shape[2],9,G3_shape[3]]),[G1_shape[2],3,3,G3_shape[3]]),[1,2,0,3])

    return value






def get_W_2d(W_4d):
    # (3,3,i,o) --> (i, 9o)
    W_2d = np.reshape(np.reshape(np.transpose(W_4d, [2,0,1,3]), [W_4d.shape[2],9,-1]),[W_4d.shape[2],-1])
    return W_2d




def get_dict():
    orig_checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'

    orig_dict = {}
    orig_reader = pywrap_tensorflow.NewCheckpointReader(orig_checkpoint_path)
    orig_var_to_shape_map = orig_reader.get_variable_to_shape_map()

    for key in orig_var_to_shape_map:
        orig_dict[key] = orig_reader.get_tensor(key)


    return orig_dict



def iter_odd(rank_rate_ave, rank_rate_single, iter_num):

    value_dict = get_dict()

    layer_nums = [2,3,4]
    repeats= [4,6,3]

    '=============================================================0初始化=========================================================='
    dict_for_saving_parameter = {}
    #零初始化
    for i in range(3):  # layer
        layer_num = i + 2

        repeat_num = 1  #1... repeats[i]-1
        conv_num = 1  # conv
        orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
        ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

        G3, G2, G1 = decompose_single(value_dict[orig_name], rank_rate_single)
        #仅分解来得到shape，从而构造相应形状的初始化零张量

        locals()[ave_name+'3'] = np.zeros(G3.shape).astype(np.float32)
        locals()[ave_name+'2'] = np.zeros(G2.shape).astype(np.float32)
        locals()[ave_name+'1'] = np.zeros(G1.shape).astype(np.float32)
        dict_for_saving_parameter[ave_name + '3'] = locals()[ave_name + '3']
        dict_for_saving_parameter[ave_name + '2'] = locals()[ave_name + '2']
        dict_for_saving_parameter[ave_name + '1'] = locals()[ave_name + '1']


    #L/2層
    for i in range(3):
        layer_num = i + 2
        orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'
        ave_name = 'layer' + str(layer_num) + '.0.conv1.weight_ave_reuse'

        G1 = locals()['layer' + str(layer_num) + '.conv1.weight_ave_reuse1']
        shape = [1,1,G1.shape[2]//2, G1.shape[3]]
        assert(G1.shape[2]//2 *2 ==G1.shape[2])

        locals()[ave_name+'1'] = np.zeros(shape).astype(np.float32)
        dict_for_saving_parameter[ave_name+'1'] = locals()[ave_name+'1']






    '=======================================================迭代==========================================='
    for iters in range(12):
        print('\n\n','='*10,'ites',iters,'='*10)



  

        '===================求独立G==================='
        for i in range(3):
            layer_num = i + 2

            for j in range(repeats[i] - 1):  # repeat
                repeat_num = j + 1  # 1... repeats[i]-1
                conv_num = 1

                orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                diff_name = orig_name + '_diff'
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

                G_ave3 = locals()[ave_name + '3']
                G_ave2 = locals()[ave_name + '2']
                G_ave1 = locals()[ave_name + '1']

                if iters==0:
                    locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name], rank_rate_single)
                else:
                    locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name]- reverse_decompose_ave(G_ave3, G_ave2, G_ave1), rank_rate_single)

                dict_for_saving_parameter[diff_name + '3'] = locals()[diff_name + '3'].astype(np.float32)
                dict_for_saving_parameter[diff_name + '2'] = locals()[diff_name + '2'].astype(np.float32)
                dict_for_saving_parameter[diff_name + '1'] = locals()[diff_name + '1'].astype(np.float32)



                '更新W_ave'
                if j==0 :  # repeat_num=1;創建變量
                    locals()[ave_name] = 0

                locals()[ave_name] += value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[ diff_name + '1'])


            if i==1:
                print(np.linalg.norm(np.reshape(locals()[diff_name + '3'],[locals()[diff_name + '3'].shape[0],-1])))
                print(np.linalg.norm(np.reshape(locals()[diff_name + '2'],[locals()[diff_name + '2'].shape[0],-1])))
                print(np.linalg.norm(np.reshape(locals()[diff_name + '1'],[locals()[diff_name + '1'].shape[0],-1])))
                print('\n')

        for i in range(3):
            layer_num = i + 2
            ave_name_common = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'

            orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'
            diff_name = orig_name + '_diff'

            G_ave3 = locals()[ave_name_common + '3']
            G_ave2 = locals()[ave_name_common + '2']

            G_ave1 = locals()[orig_name + '_ave_reuse1']

            if iters==0:
                locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name], rank_rate_single)

            else:
                locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name]- reverse_decompose_ave(G_ave3, G_ave2, G_ave1), rank_rate_single)



            dict_for_saving_parameter[diff_name + '3'] = locals()[diff_name + '3'].astype(np.float32)
            dict_for_saving_parameter[diff_name + '2'] = locals()[diff_name + '2'].astype(np.float32)
            dict_for_saving_parameter[diff_name + '1'] = locals()[diff_name + '1'].astype(np.float32)



            locals()['W' + str(layer_num) + '_ave'] = value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'])


        '=======================保存=========================='
        if iters == iter_num:
            return dict_for_saving_parameter

        #重置0
        dict_for_saving_parameter = {}


        '===================求共同G==================='

        for i in range(3):
            layer_num = i + 2
            ave_name = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'

            W_stack_2d = np.vstack((get_W_2d( locals()['W' + str(layer_num) + '_ave']), get_W_2d( locals()[ave_name])))
            G3, G2, G1, G1_first = decompose_W1(W_stack_2d, rank_rate_ave)

            locals()[ave_name + '3'] = G3
            locals()[ave_name + '2'] = G2

            '''!!!!!!!!!!!!!!!!!!!!!!/3!!!!!!!!!!!!!!'''
            locals()[ave_name + '1'] = G1 / repeats[i]

            print('\ntest')
            print(np.sum(abs(locals()[ave_name + '2'])))
            print('\n')

            dict_for_saving_parameter[ave_name + '3'] = locals()[ave_name + '3'].astype(np.float32)
            dict_for_saving_parameter[ave_name + '2'] = locals()[ave_name + '2'].astype(np.float32)
            dict_for_saving_parameter[ave_name + '1'] = locals()[ave_name + '1'].astype(np.float32)


            orig_name = 'layer' + str(layer_num) +  '.0.conv1.weight'
            locals()[orig_name + '_ave_reuse1'] = G1_first
            dict_for_saving_parameter[orig_name + '_ave_reuse1'] = locals()[orig_name + '_ave_reuse1'].astype(np.float32)

            if i==0:
                print(np.linalg.norm(np.reshape(G3,[G3.shape[0],-1])))
                print(np.linalg.norm(np.reshape(G2,[G2.shape[0],-1])))
                print(np.linalg.norm(np.reshape(G1,[G1.shape[0],-1])))
                print(np.linalg.norm(np.reshape(G1_first,[G1_first.shape[0],-1])))







# rank_rate_ave = eval(FLAGS.rank_rate_ave)
# rank_rate_single = eval(FLAGS.rank_rate_single)
# value_dict = get_dict()
# iter_odd(value_dict,rank_rate_ave,rank_rate_single)