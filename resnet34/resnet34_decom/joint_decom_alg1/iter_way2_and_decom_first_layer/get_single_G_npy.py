import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tensorflow.python import pywrap_tensorflow


def get_raw_ave():
    checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    repeat = [3, 4, 6, 3]
    ave_tenosr_collect_conv = {}

    for layer_num in range(2, 5):
        ave_tenosr_conv1 = 0
        ave_tenosr_conv2 = 0
        for repeat_num in range(1, repeat[layer_num - 1]):
            key_conv1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv1' + '.weight'
            key_conv2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv2' + '.weight'
            ave_tenosr_conv1 += reader.get_tensor(key_conv1)
            ave_tenosr_conv2 += reader.get_tensor(key_conv2)
        ave_tenosr_collect_conv['layer' + str(layer_num) + '.conv1' + '.weight_ave_reuse'] = ave_tenosr_conv1 / (repeat[layer_num - 1] - 1)
        ave_tenosr_collect_conv['layer' + str(layer_num) + '.conv2' + '.weight_ave_reuse'] = ave_tenosr_conv2 / (repeat[layer_num - 1] - 1)

    new_var_dic = {}

    for key in var_to_shape_map:
        value = reader.get_tensor(key)

        # 需要的层ave赋值替换
        if 'layer' in key and 'layer1' not in key and '.0.conv' not in key and 'Momentum' not in key and 'weight' in key and 'downsample' not in key and 'fc' not in key and 'bn' not in key:
            location = key.find('.conv')

            new_key = key[0:location - 2] + key[location:] + '_ave_reuse'
            if new_key in new_var_dic:
                assert (np.sum(ave_tenosr_collect_conv[new_key] - new_var_dic[new_key]) == 0)
            else:
                new_var_dic[new_key] = ave_tenosr_collect_conv[new_key]
            new_var_dic[key] = value
        # 其他的copy
        else:
            new_var_dic[key] = value

    print('layer2.1.conv1.weight' in var_to_shape_map)

    print('layer2.1.conv1.weight' in new_var_dic)


    repeat = [3, 4, 6, 3]
    for layer_num in range(2, 5):

        for repeat_num in range(1, repeat[layer_num - 1]):
            orig_key_conv1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv1' + '.weight'
            orig_key_conv2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv2' + '.weight'

            ave_conv1 = 'layer' + str(layer_num) + '.conv1' + '.weight_ave_reuse'
            ave_conv2 = 'layer' + str(layer_num) + '.conv2' + '.weight_ave_reuse'

            diff_conv1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv1' + '.weight_diff'
            diff_conv2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv2' + '.weight_diff'


            new_var_dic[diff_conv1] = new_var_dic[orig_key_conv1] - new_var_dic[ave_conv1]
            print(np.sum(new_var_dic[diff_conv1] -( new_var_dic[orig_key_conv1] - new_var_dic[ave_conv1])))
            new_var_dic[diff_conv2] = new_var_dic[orig_key_conv2] - new_var_dic[ave_conv2]
            print(np.sum(new_var_dic[diff_conv2] -( new_var_dic[orig_key_conv2] - new_var_dic[ave_conv2])))

    for key in new_var_dic:
        if 'weight' in key and 'ome' not in key:
            print(key)

    return new_var_dic





def dict_ave():
    # ckpt='trained_acc0.6493589743589744-8'
    #
    # path_W_ave = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave3/rerun'
    checkpoint_path = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave/7_acc0.64157-7'
    # checkpoint_path = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave/1_acc0.59872-1'


    dict = {}
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()


    for key in var_to_shape_map:
        # if 'ave' in key and 'oment' not in key:
        dict[key] = reader.get_tensor(key)
        print(key)

    # print('ave:')
    print('\n\n')
    np.save('W_ave_acc0.64157.npy')
    return dict


def dict_orig():

    checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    new_var_dic = {}

    for key in var_to_shape_map:
        # value = reader.get_tensor(key)


        new_var_dic[key] = reader.get_tensor(key)

    # np.save('parameter.npy', new_var_dic)
    return new_var_dic



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



d = dict_ave()
for key in d:
    print(key)
# new_var_dic = get_raw_ave()
# np.save('raw_ave_and_diff.npy',new_var_dic)
# test()
# orig_dict = save()
# for key in orig_dict:
#     print(key)
# np.save('Diff_and_Ave_and_Orig_low_acc_ckpt1.npy', orig_dict)



# with tf.device('/cpu:0'):
    #
    # dict = save()
    # np.save('Diff_and_Ave_and_Orig.npy', dict)
    # dict = dict_ave()

    # np.save('just_ave.npy', dict)










