import tensorflow  as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='1'



from tensorflow.python import pywrap_tensorflow


def decompose(key, value, new_var_dic):
    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    # [H,W,I,O]
    value_2d = np.reshape(value, [h * w * i, o])
    U2, S2, V2 = np.linalg.svd(value_2d)

    assert (o <= len(S2))

    # max rank: o
    V2 = np.matmul(np.diag(S2)[:o, :o], V2[:o, :])
    U2_cut = U2[:, :o]

    # to [i,h,w,o] and then [i, hwo]
    U2_cut = np.transpose(np.reshape(U2_cut, [h, w, i, o]), [2, 0, 1, 3])
    U2_cut = np.reshape(U2_cut, [i, h * w * o])

    U1, S1, V1 = np.linalg.svd(U2_cut)
    # max rank: i
    V1 = np.matmul(np.diag(S1)[:i, :i], V1[:i, :])
    U1_cut = U1[:, :i]

    G3 = np.reshape(V2, [1, 1, o, o])
    # G2 = np.reshape(np.transpose(np.reshape(V1, [i, h, w, o]), [1, 2, 0, 3]), [h, w, i, o])
    G2 = np.transpose(np.reshape(V1, [i, h, w, o]), [1, 2, 0, 3])

    G1 = np.reshape(U1_cut, [1, 1, i, i])

    new_var_dic[key] = value
    new_var_dic[key + '3'] = G3
    new_var_dic[key + '2'] = G2
    new_var_dic[key + '1'] = G1

    return new_var_dic

def save_npy():


    checkpoint_path = os.path.join("../../ckpt2", "resnet34_pytorch-10")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    new_var_dic = {}

    for key in var_to_shape_map:
        value = reader.get_tensor(key)

        if 'weight' in key and 'layer' in key and 'downsample' not in key and 'fc' not in key:
            print(key)
            print(value.shape)
            new_var_dic = decompose(key, value, new_var_dic)

        else:
            new_var_dic[key] = value

    np.save('single_tt_decom.npy', new_var_dic)

    value_dict = np.load('./single_tt_decom.npy', allow_pickle=True).item()
    print(value_dict['fc.weight'].shape)







def decompose_to_ckpt(key, value, new_var_list):
    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    # [H,W,I,O]
    value_2d = np.reshape(value, [h * w * i, o])
    U2, S2, V2 = np.linalg.svd(value_2d)

    assert (o <= len(S2))

    # max rank: o
    V2 = np.matmul(np.diag(S2)[:o, :o], V2[:o, :])
    U2_cut = U2[:, :o]

    # to [i,h,w,o] and then [i, hwo]
    U2_cut = np.transpose(np.reshape(U2_cut, [h, w, i, o]), [2, 0, 1, 3])
    U2_cut = np.reshape(U2_cut, [i, h * w * o])

    U1, S1, V1 = np.linalg.svd(U2_cut)
    # max rank: i
    V1 = np.matmul(np.diag(S1)[:i, :i], V1[:i, :])
    U1_cut = U1[:, :i]

    G3 = np.reshape(V2, [1, 1, o, o])
    G2 = np.reshape(np.transpose(np.reshape(V1, [i, h, w, o]), [1, 2, 0, 3]), [h, w, i, o])
    G1 = np.reshape(U1_cut, [1, 1, i, i])

    G3 = tf.Variable(G3, name= key + '3')
    G2 = tf.Variable(G2, name= key + '2')
    G1 = tf.Variable(G1, name= key + '1')


    new_var_list.append(G3)
    new_var_list.append(G2)
    new_var_list.append(G1)


    return new_var_list





checkpoint_path = "/home/test01/csw/resnet8/ckpt_250epoch_another_bn_api/resnet8-acc0.93009-250"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

new_var_dic = {}

for key in var_to_shape_map:
    value = reader.get_tensor(key)

    new_var_dic[key] = value

np.save('parameter.npy', new_var_dic)



