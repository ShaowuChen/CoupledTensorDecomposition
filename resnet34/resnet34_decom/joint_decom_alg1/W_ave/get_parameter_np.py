import tensorflow  as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'


from tensorflow.python import pywrap_tensorflow






def save_npy():

    checkpoint_path = '/home/test01/csw/resnet34/ckpt_with_relu_input/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'

    # checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()



    repeat = [3,4,6,3]
    ave_tenosr_collect_conv = {}

    for layer_num in range(2,5):
        ave_tenosr_conv1 = 0
        ave_tenosr_conv2 = 0
        for repeat_num in range(1,repeat[layer_num-1]):
            key_conv1 = 'layer'+str(layer_num)+'.'+str(repeat_num)+'.'+'conv1'+'.weight'
            key_conv2 = 'layer'+str(layer_num)+'.'+str(repeat_num)+'.'+'conv2'+'.weight'
            ave_tenosr_conv1 +=  reader.get_tensor(key_conv1)
            ave_tenosr_conv2 +=  reader.get_tensor(key_conv2)
        ave_tenosr_collect_conv['layer'+str(layer_num)+'.conv1'+'.weight_ave_reuse'] = ave_tenosr_conv1/(repeat[layer_num-1]-1)
        ave_tenosr_collect_conv['layer'+str(layer_num)+'.conv2'+'.weight_ave_reuse'] = ave_tenosr_conv2/(repeat[layer_num-1]-1)


    new_var_dic = {}

    for key in var_to_shape_map:
        value = reader.get_tensor(key)

        #需要的层ave赋值替换
        if 'layer' in key and 'layer1' not in key  and '.0.conv' not in key and 'Momentum' not in key and 'weight' in key and 'downsample' not in key and 'fc' not in key and 'bn' not in key:
            location = key.find('.conv')
            
            new_key = key[0:location-2] + key[location:]+'_ave_reuse'
            if new_key in new_var_dic:
                assert(np.sum(ave_tenosr_collect_conv[new_key]-new_var_dic[new_key])==0)
            else:
                new_var_dic[new_key] = ave_tenosr_collect_conv[new_key]

        #其他的copy
        else:
            new_var_dic[key] = value

    for key in new_var_dic:
        if 'weight' in key and 'oment' not in key:
            print(key)

    np.save('joint_ave_with_relu.npy', new_var_dic)

    # value_dict = np.load('./single_tt_decom.npy', allow_pickle=True).item()
    # print(value_dict['fc.weight'].shape)


def save_npy_orig(dict):
    checkpoint_path = os.path.join("/home/test01/csw/resnet34/ckpt2", "resnet34_pytorch-10")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        dict[key] = reader.get_tensor(key)
        dict[key] = reader.get_tensor(key)

    return dict




def save_npy_ave():
    checkpoint_path = os.path.join("/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave", "trained_acc0.64453125-5")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()

    ave_tenosr_collect_conv = {}

    for key in var_to_shape_map:

        if 'weight_ave_reuse' in key and 'layer1' not in key:
            ave_tenosr_collect_conv[key] = reader.get_tensor(key)

    return ave_tenosr_collect_conv



save_npy()

# dict = save_npy_orig(save_npy_ave())
# for key in dict:
#     if 'ave' in key and 'Momen' not in key:
#         print(key)
#
# np.save('joint_ave_20200616.npy', dict)