# # import tensorflow.compat.v1 as tf
# # import tensorflow as tf
import torch
import numpy as np
# from tensorflow.python import pywrap_tensorflow
# from tensorflow.python.platform import gfile


# def convert(bin_path, ckptpath):
#     with tf.Session() as sess:
#         for var_name, value in torch.load(bin_path, map_location='gpu').items():
#             tf.Variable(initial_value=value.data.numpy(), name=var_name)
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#         saver.save(sess, ckpt_path)

bin_path = 'E:/谢敏/resnet50-19c8e357.pth'
ckpt_path = 'bert_model.ckpt'

dictionary = {}
# with tf.Session() as sess:
i=0
for var_name, value in torch.load(bin_path).items():


    numpy_value = value.data.numpy()
    if len(numpy_value.shape)==4:
        numpy_value = np.transpose(numpy_value, [2,3,1,0])
    dictionary[var_name] = numpy_value
print(type(dictionary))

np.save('E:/谢敏/resnet50.npy', dictionary)
# data_dict = np.load('./resnet34.npy', allow_pickle=True).item()
#
# print(data_dict['layer1.0.conv1.weight'].shape)

# i=0
# for f in torch.load(bin_path):
#     dictionary[f] = 


