# import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow

def get_parameter_dict():
    # ckpt='trained_acc0.6493589743589744-8'
    #
    # path_W_ave = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave3/rerun'
    checkpoint_path = '/home/test01/csw/resnet10/not_shorcut/ckpt_stddev0.01/resnet10-acc0.69369-75'
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
    # np.save('W_ave_acc0.64157.npy')
    return dict

get_parameter_dict()