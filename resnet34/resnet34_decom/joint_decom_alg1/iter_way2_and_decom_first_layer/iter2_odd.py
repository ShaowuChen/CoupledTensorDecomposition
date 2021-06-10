import tensorflow as tf
import numpy as np

import os
from tensorflow.python import pywrap_tensorflow

flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('gpu', '0', 'rank_rate of each block1 conv2')
flags.DEFINE_string('ckpt_path', '20200725', 'rank_rate of each block1 conv2')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]
os.makedirs(ckpt_path,exist_ok=True)

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
    print('rank1:',rank1)

    # [H,W,I,O] -->  [I, H, W, O]  --> [I, HWO]
    value_t = np.transpose(value, [2,0,1,3])
    value_2d = np.reshape(value_t, [i, h*w*o])


    G1_2d,S1, G23_2d = np.linalg.svd(value_2d)
    G1_2d = G1_2d[:,:rank1]  #[i, rank1]
    G23_2d = np.matmul(np.diag(S1)[:rank1, :rank1], G23_2d[:rank1,:])  #[rank1, hwo]


    assert (G1_2d.shape == (i, rank1))
    G1 = np.reshape(G1_2d, [1,1,i,rank1])

    G23_2d = np.reshape( np.reshape(G23_2d, [rank1,h,w,o]), [rank1*h*w,o])
    G2_2d, S2, G3_2d = np.linalg.svd(G23_2d)
    G2_2d = G2_2d[:,:rank2]
    G3_2d = np.matmul(np.diag(S2)[:rank2, :rank2], G3_2d[:rank2,:])

    assert(G2_2d.shape == (rank1*h*w, rank2))
    G2 = np.transpose( np.reshape(G2_2d, [rank1, h, w, rank2]), [1,2,0,3])


    assert(G3_2d.shape == (rank2, o))
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
    weight = np.transpose( np.reshape(weight_2d, [i, 3, 3, o ]), [1,2,0,3])

    return weight


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

    rank2 = int(np.floor(i*rank_rate))
    rank1 = min(rank2, i)
    assert(rank1==rank2)

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

    V1 = np.reshape( np.transpose(G_ave2,[2,0,1,3]), [G_ave2_shape[3],-1])


    U2_temp = np.matmul(U1,V1)
    U2 = np.reshape(np.transpose( np.reshape(U2_temp,[I,H,W,-1]), [1,2,0,3]), [H*W*I, -1])

    V2 = np.reshape(G_ave3, [-1,O])

    weight_reverse_2d = np.matmul(U2,V2)

    assert(weight_reverse_2d.shape==(H*W*I,O))
    weight_reverse_4d = np.reshape(weight_reverse_2d, [H,W,I,O])
    return weight_reverse_4d


def get_G23_2d(G2, G3):
    #G2: [3,3,r1,r2] --> [9r1, r2]
    #G3: [1,1,r2,o]  --> [r2, o]
    #--> G23 [9r1, o]  ---> [r1, 9o]
    G23 = np.matmul( np.reshape(G2, [-1, G2.shape[-1]]), np.reshape(G3, [G3.shape[2], G3.shape[3]]) )
    assert(G23.shape == (9*G2.shape[2], G3.shape[3]))
    G23 = np.reshape(np.transpose(np.reshape(G23, [3, 3, G2.shape[2], G3.shape[3]]), [2,0,1,3]), [G2.shape[2], 9*G3.shape[3]])
    return G23

def decompose_G23_2d(G23_2d, rank_rate_ave):
    #[r, 9o]
    r1 = G23_2d.shape[0]
    assert(G23_2d.shape[1]%9==0)
    o = G23_2d.shape[1]//9
    G23_t = np.reshape(np.reshape(G23_2d, [r1, 3,3,o]), [r1*9, o])

    G2_2d,S,G3_2d = np.linalg.svd(G23_t)

    rank = int(np.floor(rank_rate_ave*o))
    assert (rank <= len(S))
    G2_4d = np.transpose( np.reshape( np.matmul( G2_2d[:, :rank], np.diag(S)[:rank,:rank]), [r1,3,3,rank]), [1,2,0,3])

    G3_2d = G3_2d[:rank, :]
    assert(G3_2d.shape == (rank, o))
    G3_4d = np.reshape( G3_2d, [1,1,rank,o])

    return G3_4d, G2_4d


#求偽逆
def pinv_G23_2d(G23_2d):
    #[i, 9o]
    return np.matmul(np.linalg.inv(np.matmul(G23_2d.T, G23_2d)), G23_2d.T)

def get_W_2d(W_4d):
    # (3,3,i,o) --> (i, 9o)
    W_2d = np.reshape(np.transpose(W_4d, [2,0,1,3]), [W_4d.shape[2],-1])
    return W_2d

def get_G1_2d(G1_4d):
    # (1,1,i,o) --> (i, o)

    G1_2d = np.reshape(G1_4d, [G1_4d.shape[2],G1_4d.shape[3]])
    return G1_2d


def get_dict():
    orig_checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'

    orig_dict = {}
    orig_reader = pywrap_tensorflow.NewCheckpointReader(orig_checkpoint_path)
    orig_var_to_shape_map = orig_reader.get_variable_to_shape_map()

    for key in orig_var_to_shape_map:
        orig_dict[key] = orig_reader.get_tensor(key)


    return orig_dict

def get_norm(tensor):
    return np.linalg.norm(np.reshape(tensor,[-1,tensor.shape[3]]))

def norm_recovery(orig_norm, G3, G2, G1):
    W = reverse_decompose_odd(G3,G2,G1)
    now_norm = np.linalg.norm(np.reshape(W, [-1, W.shape[3]]))
    G3 = G3 * orig_norm / now_norm
    G2 = G2 * orig_norm / now_norm
    G1 = G1 * orig_norm / now_norm

    print('\n')
    print(orig_norm)
    print(now_norm)
    W = reverse_decompose_odd(G3,G2,G1)
    now_norm = np.linalg.norm(np.reshape(W, [-1, W.shape[3]]))
    print(now_norm)
    print('\n')

    return G3, G2, G1


def iter_odd(value_dict, rank_rate_ave, rank_rate_single):

    layer_nums = [2,3,4]
    repeats= [4,6,3]

    '=============================================================初始化=========================================================='
    #初始化，GEGE獨立的G
    #初始化，得到 W_ave 分解的G1， G2， G3
    for i in range(3):  # layer
        layer_num = i + 2

        for j in range(repeats[i] - 1):  # repeat
            repeat_num = j + 1  #1... repeats[i]-1
            for conv_num in [1]:  # conv
                orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                diff_name = orig_name + '_diff'
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

                if ave_name not in locals():  #j=1;創建變量
                    locals()[ave_name] = 0

                locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name], rank_rate_single)

                locals()[ave_name] += (value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1']))/(repeats[i]-1)
                if repeat_num == repeats[i] - 1:

                    locals()[ave_name+'3'], locals()[ave_name+'2'], locals()[ave_name+'1'] = decompose_odd(locals()[ave_name], rank_rate_ave)
                    locals()[ave_name + '3norm'] = get_norm(locals()[ave_name + '3'])
                    locals()[ave_name + '2norm'] = get_norm(locals()[ave_name + '2'])
                    locals()[ave_name + '1norm'] = get_norm(locals()[ave_name + '1'])
        if i==2:
            print(locals()[ave_name + '3'][0, 0, 0, 0])

    #初始化，得到每個L/2層對應的G1
    for i in range(3):
        layer_num = i + 2
        orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'
        diff_name = orig_name + '_diff'
        ave_name = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'

        locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name], rank_rate_single)
        locals()['W'+str(layer_num)+'_ave'] = value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'])

        locals()[ave_name + '_G23_2d'] = get_G23_2d(locals()[ave_name + '2'], locals()[ave_name + '3'])
        '''L/2層 G1特殊的命名規則'''
        temp = np.matmul(pinv_G23_2d(locals()[ave_name + '_G23_2d'].T) , get_W_2d(locals()['W'+str(layer_num)+'_ave']).T).T
        locals()[orig_name + '_ave_reuse1'] = np.reshape( temp, [1,1,temp.shape[0], temp.shape[1]])
        locals()[orig_name + '_ave_reuse1norm'] = get_norm(locals()[orig_name + '_ave_reuse1'])




    '=======================================================開始迭代==========================================='
    for iters in range(12):
        print('='*20)
        print('ites',iters)


        dict_for_saving_parameter = {}

        '=======================保存=========================='
        #普通層
        for i in range(3):  # layer
            layer_num = i + 2

            for j in range(repeats[i] - 1):  # repeat
                repeat_num = j + 1  # 1... repeats[i]-1
                conv_num = 1

                orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
                diff_name = orig_name + '_diff'
                ave_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'

                dict_for_saving_parameter[ave_name + '1'] = locals()[ave_name + '1']
                dict_for_saving_parameter[ave_name + '2'] = locals()[ave_name + '2']
                dict_for_saving_parameter[ave_name + '3'] = locals()[ave_name + '3']

                dict_for_saving_parameter[diff_name + '1'] = locals()[diff_name + '1']
                dict_for_saving_parameter[diff_name + '2'] = locals()[diff_name + '2']
                dict_for_saving_parameter[diff_name + '3'] = locals()[diff_name + '3']

        #L/2層
        for i in range(3):
            layer_num = i + 2
            orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'
            diff_name = orig_name + '_diff'

            dict_for_saving_parameter[orig_name + '_ave_reuse1'] = locals()[orig_name + '_ave_reuse1']

            dict_for_saving_parameter[diff_name + '1'] = locals()[diff_name + '1']
            dict_for_saving_parameter[diff_name + '2'] = locals()[diff_name + '2']
            dict_for_saving_parameter[diff_name + '3'] = locals()[diff_name + '3']

        '一個iter結束；保存'
        if iters == 0 or iters == 3 or iters == 5 or iters == 10:
            np.save(ckpt_path + '/iter_odd' + str(iters) + '.npy', dict_for_saving_parameter)
            # pass




        '=======================================更新獨立G==============================='
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
                locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name]- reverse_decompose_odd(G_ave3, G_ave2, G_ave1), rank_rate_single)

                '更新W_ave'
                # if i==0 :  # j=1;創建變量
                '''修改'''
                if j==0:
                    locals()[ave_name] = 0


                locals()[ave_name] += (value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[ diff_name + '1'])) / (repeats[i] - 1)

        for i in range(3):
            layer_num = i + 2
            ave_name_common = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'

            orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'
            diff_name = orig_name + '_diff'

            G_ave3 = locals()[ave_name_common + '3']
            G_ave2 = locals()[ave_name_common + '2']
            G_ave1 = locals()[orig_name + '_ave_reuse1']



            locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'] = decompose_single(value_dict[orig_name]- reverse_decompose_odd(G_ave3, G_ave2, G_ave1), rank_rate_single)

            locals()['W' + str(layer_num) + '_ave'] = value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'])


        '==================================更新G2，G3=========================='

        for i in range(3):
            layer_num = i + 2
            ave_name = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'
            orig_name = 'layer' + str(layer_num) + '.0.conv1.weight'

            W_stack_2d = np.vstack((get_W_2d( locals()['W' + str(layer_num) + '_ave']), get_W_2d( (repeats[i]-1)*locals()[ave_name])))

            assert(len(locals()[orig_name+'_ave_reuse1'].shape)==4)
            assert(len(locals()[ave_name+'1'].shape)==4)

            G_stack_2d = np.vstack( (get_G1_2d(locals()[orig_name +'_ave_reuse1']),   get_G1_2d((repeats[i]-1)*locals()[ave_name +'1'])))



            #[r, 9o]
            G23_2d = (np.matmul(W_stack_2d.T, np.matmul(np.linalg.inv(np.matmul(G_stack_2d, G_stack_2d.T)), G_stack_2d)) ).T

            locals()[ave_name + '3'], locals()[ave_name + '2'] = decompose_G23_2d(G23_2d, rank_rate_ave)

            #能量恢復
            locals()[ave_name + '3'] = locals()[ave_name + '3'] * locals()[ave_name + '3norm'] / get_norm(locals()[ave_name + '3'])
            locals()[ave_name + '2'] = locals()[ave_name + '2'] * locals()[ave_name + '2norm'] / get_norm(locals()[ave_name + '2'])




            if i==2:
                print('\n G2' )
                print(np.linalg.norm(np.reshape(locals()[ave_name + '2'],[-1,9])),'\n')

                print('G3' )
                print(np.linalg.norm(np.reshape(locals()[ave_name + '3'],[-1,locals()[ave_name + '3'].shape[3]])),'\n\n')
                print(locals()[ave_name + '3'][0,0,0,0])


        '===================================更新G1, G1_2/L==========================================='
        for i in range(3):
            layer_num = i + 2

            #with G2_ave, G3_ave, get G1_ave
            ave_name = 'layer' + str(layer_num) + '.conv1.weight_ave_reuse'
            locals()[ave_name + '_G23_2d'] = get_G23_2d(locals()[ave_name+'2'], locals()[ave_name+'3'])
            temp = np.matmul(pinv_G23_2d(locals()[ave_name + '_G23_2d'].T), get_W_2d(locals()[ave_name]).T).T
            locals()[ave_name + '1'] = np.reshape( temp, [1,1,temp.shape[0], temp.shape[1]])
            locals()[ave_name + '1'] = locals()[ave_name + '1'] * locals()[ave_name + '1norm'] / get_norm(locals()[ave_name + '1'])




            if i==2:
                print('\nG1')
                print(np.linalg.norm(np.reshape(locals()[ave_name + '1'],(-1,locals()[ave_name + '1'].shape[2]))),'\n')

            #with G2_ave, G3_ave,get G1_first_conv
            orig_name = 'layer' + str(layer_num) +  '.0.conv1.weight'
            diff_name = orig_name + '_diff'
            '''L/2層 G1特殊的命名規則'''
            locals()['W' + str(layer_num) + '_ave'] = value_dict[orig_name] - reverse_decompose_single(locals()[diff_name + '3'], locals()[diff_name + '2'], locals()[diff_name + '1'])
            locals()[ave_name + '_G23_2d'] = get_G23_2d(locals()[ave_name + '2'], locals()[ave_name + '3'])
            temp = np.matmul(pinv_G23_2d(locals()[ave_name + '_G23_2d'].T) , get_W_2d(locals()['W'+str(layer_num)+'_ave']).T).T
            locals()[orig_name + '_ave_reuse1'] = np.reshape( temp, [1,1,temp.shape[0], temp.shape[1]])

            locals()[orig_name + '_ave_reuse1'] = locals()[orig_name + '_ave_reuse1'] * locals()[orig_name + '_ave_reuse1norm'] / get_norm(locals()[orig_name + '_ave_reuse1'])

            if i==2:
                print('\nG1\'')
                print(np.linalg.norm(np.reshape(locals()[orig_name + '_ave_reuse1'],[-1,locals()[orig_name + '_ave_reuse1'].shape[2]])),'\n')



rank_rate_ave = eval(FLAGS.rank_rate_ave)
rank_rate_single = eval(FLAGS.rank_rate_single)
value_dict = get_dict()
iter_odd(value_dict,rank_rate_ave,rank_rate_single)