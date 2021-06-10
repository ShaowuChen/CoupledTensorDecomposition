import tensorflow  as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='7'



from tensorflow.python import pywrap_tensorflow



# checkpoint_path = os.path.join("../../ckpt2", "resnet34_pytorch-10")
checkpoint_path = '/home/test01/csw/resnet8/ckpt_250epoch_another_bn_api/resnet8-acc0.92070-100'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

new_var_dic = {}

for key in var_to_shape_map:
    # value = reader.get_tensor(key)

    # new_var_dic[key] = value
    # if 'down' in key:
        # print(key)
    print(key)


# np.save('./parameter.npy', new_var_dic)



