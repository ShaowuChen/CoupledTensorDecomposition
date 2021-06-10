import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


from tensorflow.python import pywrap_tensorflow

def dict_ave():
    ckpt='trained_acc0.6493589743589744-8'

    path_W_ave = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave3/rerun'
    checkpoint_path = os.path.join(path_W_ave, ckpt)

    dict = {}
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()


    for key in var_to_shape_map:
        # if 'ave' in key and 'oment' not in key:
        dict[key] = reader.get_tensor(key)
        # print(key)

    # print('ave:')
    print('\n\n')
    return dict


def dict_orig():

    checkpoint_path = os.path.join("/home/test01/csw/resnet34/ckpt2", "resnet34_pytorch-10")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    new_var_dic = {}

    for key in var_to_shape_map:
        value = reader.get_tensor(key)


        new_var_dic[key] = value

    # np.save('parameter.npy', new_var_dic)
    return new_var_dic

with tf.device('/cpu:0'):

    dict_ave = dict_ave()
    dict_orig = dict_orig()
    for key in dict_orig:

        if 'bn' in key and 'omen' not in key:
            print(key)
            print(np.sum(abs(dict_ave[key] - dict_orig[key])))

    print('\n\n')
    for key in dict_orig:

        if 'down' in key and 'omen' not in key:
            print(key)
            print(np.sum(abs(dict_ave[key] - dict_orig[key])))


def save():

    ave_dict = dict_ave()
    orig_dict = dict_orig()


    repeat = [3, 4, 6, 3]
    for layer_num in range(2, 5):

        for repeat_num in range(1, repeat[layer_num - 1]):
            orig_key_conv1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv1' + '.weight'
            orig_key_conv2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv2' + '.weight'

            ave_conv1 = 'layer' + str(layer_num) + '.conv1' + '.weight_ave_reuse'
            ave_conv2 = 'layer' + str(layer_num) + '.conv2' + '.weight_ave_reuse'

            diff_conv1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv1' + '.weight_diff'
            diff_conv2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv2' + '.weight_diff'

            orig_dict[diff_conv1] = orig_dict[orig_key_conv1] - ave_dict[ave_conv1]
            # print(np.sum(orig_dict[diff_conv1] -( orig_dict[orig_key_conv1] - ave_dict[ave_conv1])))
            orig_dict[diff_conv2] = orig_dict[orig_key_conv2] - ave_dict[ave_conv2]
            # print(np.sum(orig_dict[diff_conv2] -( orig_dict[orig_key_conv2] - ave_dict[ave_conv2])))


    orig_dict.update(ave_dict)
    return orig_dict


# orig_dict = save()
# for key in orig_dict:
#     print(key)
# np.save('Diff_and_Ave_and_Orig.npy', orig_dict)
# with tf.device('/cpu:0'):
#
#     # dict = save()
#     # np.save('Diff_and_Ave_and_Orig.npy', dict)
#     dict = dict_ave()
#     np.save('just_ave.npy', dict)










