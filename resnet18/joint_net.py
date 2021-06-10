
'''
修改日志：
2020/6/22 将 W_diff= W原 - W_ave 改为 W_diff = W原 - reverse(G_ave1,G_ave2,G_ave3), ckpt为ckpt_finetuned_ave_20200622_lr0.01


'''


import numpy as np
import os
import tensorflow as tf
import cifar10_input

flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '0.5', 'how many train_set used to train')
flags.DEFINE_string('epoch', '8', 'train epoch' )

flags.DEFINE_string('rank_rate_yellow', None, 'rank_rate of each block1 conv2')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")
flags.DEFINE_string('whole_ave', 'False', 'deecompose W_ave or not')

flags.DEFINE_string('gpu', '7', 'gpu choosed to used' )
r1=1/16
c1=0.24088485
c2=0.24088485

# if eval(FLAGS.whole_ave):
#     ckpt_path = './ckpt_finetuned0.64_ave_20200622_lr0.01/whole_ave/single_rate'+str(FLAGS.rank_rate_single[0])+'_'+str(FLAGS.rank_rate_single[2])
# else:
#     ckpt_path = './ckpt_finetuned0.64_ave_20200622_lr0.01/Both_decom/yellow_rate'+str(FLAGS.rank_rate_yellow[0])+'_'+str(FLAGS.rank_rate_yellow[2])+'_ave_rate'+str(FLAGS.rank_rate_ave[0])+'_'+str(FLAGS.rank_rate_ave[2])
# os.makedirs(ckpt_path, exist_ok=True)

if FLAGS.gpu=='all':
    pass
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存

import sys

#获取G1 G2 G3
def decompose(value, rank_rate):
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

def reverse_decompose(G_ave3,G_ave2,G_ave1):
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
    print(weight_reverse_2d.shape)
    print(H*W*I)
    print(O)
    assert(weight_reverse_2d.shape==(H*W*I,O))
    weight_reverse_4d = np.reshape(weight_reverse_2d, [H,W,I,O])
    return weight_reverse_4d

def basic_conv_ave(in_tensor, value_dict, layer_num, repeat_num,conv_num, strides):

    name_weight_ave = 'layer' + str(layer_num) + '.conv' +str(conv_num)+'.weight_ave_reuse'

    name_weight_ave1 = name_weight_ave + '1'
    name_weight_ave2 = name_weight_ave + '2'
    name_weight_ave3 = name_weight_ave + '3'

    # G3, G2, G1 = decompose(value_dict[name_weight_ave], rank_rate_ave)
    # 1层卷积变3层 (Weight [H,W,I,O])
    # print('basic_conv_ave1' + str(type(G3)))

    weight_ave1 = tf.get_variable(name=name_weight_ave1, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.aver.com'][0]),regularizer=regularizer) #共享

    tensor_go_next = tf.nn.conv2d(in_tensor, weight_ave1, [1, 1, 1, 1], padding='SAME')

    weight_ave2 = tf.get_variable(name=name_weight_ave2, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.aver.com'][1]),regularizer=regularizer) #共享
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight_ave2, [1, strides[1], strides[2], 1], padding='SAME')

    weight_ave3= tf.get_variable(name=name_weight_ave3, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.aver.com'][2]),regularizer=regularizer) #共享
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight_ave3, [1, 1, 1, 1], padding='SAME')

    # 加入训练list;冻结其他一切层
    var_list_to_train.append(weight_ave1)
    var_list_to_train.append(weight_ave2)
    var_list_to_train.append(weight_ave3)
    # print('basic_conv_ave2' + str(type(G3)))

    return tensor_go_next

def basic_conv_single(in_tensor, value_dict, layer_num, repeat_num, conv_num,strides):
    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'

    # name_weight_orig = name_scope + 'conv' + str(conv_num) + '.weight'
    name_weight = name_scope + 'conv' + str(conv_num) + '.weight_diff'
    # rank_rate = rank_rate_single
    # value_diff = value_dict[name_weight_orig] - reverse_decompose(G_ave3,G_ave2,G_ave1)
    # G3,G2,G1 = decompose(value_diff, rank_rate)


    name_weight1 = name_weight + '1'
    name_weight2 = name_weight + '2'
    name_weight3 = name_weight + '3'
    # 1层卷积变3层 (Weight [H,W,I,O])
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv2' + '.weight.diff.com'][0]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')
    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv2' + '.weight.diff.com'][1]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')
    weight3 = tf.get_variable(name=name_weight3, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv2' + '.weight.diff.com'][2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')
    #加入训练list;冻结其他一切层
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)
    var_list_to_train.append(weight3)

    return tensor_go_next

def basic_conv_f1(in_tensor, value_dict, layer_num, repeat_num, conv_num,strides):
    name_weight_s = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.f1_QH_reuse'
    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + 'conv' + str(conv_num) + '.f1_diff'
    name_weight1 = name_scope + 'P'
    name_weight2 = name_weight_s + 'Q'
    name_weight3 = name_weight_s + 'H'
    # 1层卷积变3层 (Weight [H,W,I,O])
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.f1.com'][repeat_num]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')
    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.f1.com'][2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')
    weight3 = tf.get_variable(name=name_weight3, dtype=tf.float32, initializer=tf.constant(value_dict['layer'+str(layer_num)+'.f1.com'][3]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')
    #加入训练list;冻结其他一切层
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)
    var_list_to_train.append(weight3)
    return tensor_go_next

def basic_conv_f3(in_tensor, value_dict, layer_num, repeat_num, conv_num,strides):
    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'

    # name_weight_orig = name_scope + 'conv' + str(conv_num) + '.weight'
    name_weight = name_scope + 'conv' + str(conv_num) + '.f3'
    # rank_rate = rank_rate_single
    # value_diff = value_dict[name_weight_orig] - reverse_decompose(G_ave3,G_ave2,G_ave1)
    # G3,G2,G1 = decompose(value_diff, rank_rate)


    name_weight1 = name_weight + '1'
    name_weight2 = name_weight + '2'
    name_weight3 = name_weight + '3'
    # 1层卷积变3层 (Weight [H,W,I,O])
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv1' + '.weight.f3.com'][0]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')
    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv1' + '.weight.f3.com'][1]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')
    weight3 = tf.get_variable(name=name_weight3, dtype=tf.float32, initializer=tf.constant(value_dict[name_scope+'conv1' + '.weight.f3.com'][2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')
    #加入训练list;冻结其他一切层
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)
    var_list_to_train.append(weight3)

    return tensor_go_next


def basic_conv(in_tensor, value_dict,aver_dict,single_dict,f1_dict,f3_dict, layer_num, repeat_num, conv_num,flag_shortcut, down_sample, is_training, res_in_tensor):

    print('layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv' + str(conv_num))
    print('lay_num, repeat_num:', layer_num, ' ', repeat_num)
    print('shape of in_tesnor', in_tensor.shape)
    print('\n')

    print('debug0')
    if layer_num != 1 and (repeat_num == 0 and conv_num == 1):
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]
    print('debug1')

    # 仅有layer1不分解;
    if layer_num==1 :
        name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'  + 'conv' + str(conv_num) + '.weight'
        weight = tf.Variable(value_dict[name_weight], name=name_weight)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')


    # else:
    #     if layer_num!=1 and repeat_num == 0 and conv_num == 2:#除layer1外， 其余layer的block1的第二层单独分解
    #         tensor_go_next = basic_conv_single(in_tensor, value_dict, layer_num, repeat_num, conv_num,strides=strides, rank_rate_single=rank_rate_yellow)
    else:
        if conv_num==1:   #对除了layer1的其余层，每个block的conv1层采用algorithm2方法分解

            tensor_from_f1= basic_conv_f1(in_tensor, f1_dict, layer_num, repeat_num,conv_num,strides=strides)
            # print('basic_conv'+str(type(G_ave3)))
            tensor_from_f3 = basic_conv_f3(in_tensor, f3_dict, layer_num, repeat_num, conv_num,strides=strides)

            tensor_go_next = tensor_from_f1 + tensor_from_f3
        else:#对除了layer1的其余层，每个block的conv2层采用algorithm1方法分解
            tensor_from_ave= basic_conv_ave(in_tensor, aver_dict, layer_num, repeat_num,conv_num,strides=strides)
            # print('basic_conv' + str(type(G_ave3)))
            tensor_from_single = basic_conv_single(in_tensor, single_dict, layer_num, repeat_num, conv_num,strides=strides)
            tensor_go_next = tensor_from_single + tensor_from_ave
    # tensor_go_next = tensor_from_ave


    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)
    # BN和Relu、shorcut照旧
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True, scale=True, epsilon=1e-9,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  is_training=is_training, scope=name_bn_scope)
    print('shape of tensor_go_next', tensor_go_next.shape)


    if flag_shortcut == True:

        if down_sample == True:
            name_downsample_weight = name_scope + 'downsample.0.weight'
            name_downsample_bn_scope = name_scope + 'downsample.1'

            weight_down_sample = tf.Variable(value_dict[name_downsample_weight], name=name_downsample_weight)
            shorcut_tensor = tf.nn.conv2d(res_in_tensor, weight_down_sample, [1, 2, 2, 1], padding='SAME')
            shorcut_tensor = tf.contrib.layers.batch_norm(shorcut_tensor, decay=0.9, center=True, scale=True,
                                                          epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                          is_training=is_training, scope=name_downsample_bn_scope)
        else:
            print('debug2')
            shorcut_tensor = res_in_tensor

        tensor_go_next = tensor_go_next + shorcut_tensor
    tensor_go_next = tf.nn.relu(tensor_go_next)
    print('done\n\n')
    return tensor_go_next

def basic_block(in_tensor, value_dict,aver_dict,single_dict,f1_dict,f3_dict, layer_num, repeat_num, down_sample=False, is_training=False):


    tensor_go_next =  basic_conv(in_tensor, value_dict,aver_dict,single_dict,f1_dict,f3_dict, layer_num, repeat_num, conv_num=1,is_training=is_training,flag_shortcut=False, down_sample=False,  res_in_tensor= None)
    tensor_go_next =  basic_conv(tensor_go_next, value_dict,aver_dict,single_dict,f1_dict,f3_dict, layer_num, repeat_num, conv_num=2,is_training=is_training, flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor)
    return tensor_go_next

def repeat_basic_block(in_tensor, value_dict,aver_dict,single_dict,f1_dict,f3_dict, repeat_times, layer_num, is_training):

    tensor_go_next = in_tensor
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for i in range(repeat_times):
            down_sample = True if i == 0 and layer_num != 1 else False
            tensor_go_next = basic_block(tensor_go_next, value_dict, aver_dict,single_dict,f1_dict,f3_dict,layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training)
    return tensor_go_next

def resnet18(x, value_dict, aver_dict,single_dict,f1_dict,f3_dict,is_training):



    weight_layer1 =  tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,strides=[1,1,1,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next,  decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    #resnet34忘了加relu
    tensor_go_next = tf.nn.relu(tensor_go_next)
    '''64 block1不分解'''
    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,aver_dict,single_dict,f1_dict,f3_dict, 2, 1, is_training)
    print('\nblock1 finished')
    print(tensor_go_next.shape)

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,aver_dict,single_dict,f1_dict,f3_dict, 2, 2, is_training)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,aver_dict,single_dict,f1_dict,f3_dict, 2, 3, is_training)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,aver_dict,single_dict,f1_dict,f3_dict, 2, 4, is_training)
    print('\nblock4 finished')


    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=4, strides=1)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten, weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output



regularizer = None
var_list_to_train=[]
value_dict = np.load('./parameter.npy', allow_pickle=True).item()
aver_dict=np.load('./algrithm1_para_npy/iter9_aver_compressed_dict_r1=%8f_c1=%8f.npy'%(r1,c1), allow_pickle=True).item()
single_dict=np.load('./algrithm1_para_npy/iter9_diff_compressed_dict_r1=%8f_c1=%8f.npy'%(r1,c1), allow_pickle=True).item()
f1_dict=np.load('./algrithm2_npy/iter9_f1_compressed_dict_r1=%8f_c2=%8f.npy'%(r1,c2), allow_pickle=True).item()
f3_dict=np.load('./algrithm2_npy/iter9_f3_compressed_dict_r1=%8f_c2=%8f.npy'%(r1,c2), allow_pickle=True).item()
with tf.device('/cpu:0'):
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir='./cifar10/cifar-10-batches-bin', batch_size= 256)
    test_images, test_labels = cifar10_input.inputs(data_dir='./cifar10/cifar-10-batches-bin', eval_data=True, batch_size=256)


'''Network'''
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
is_training = tf.placeholder(tf.bool,name='is_training')
x =  tf.placeholder(tf.float32, [None,32,32,3],name='x')
y =  tf.placeholder(tf.int64, [None],name='y')
lr = tf.placeholder(tf.float32, name='learning_rate')
#
# initializer = tf.truncated_normal_initializer(stddev=0.01)
# regularizer = tf.contrib.layers.l2_regularizer(5e-4) #L2 Regularizer

print('start building network')
tensor_go_next_fc, output = resnet18(x, value_dict, aver_dict,single_dict,f1_dict,f3_dict,is_training )
print('building network done\n\n')

# keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss') + tf.add_n(keys)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')
optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )
train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     train_op = optimizer.minimize(loss,name='train_op')


correct_predict = tf.equal(tf.argmax(output, 1), y, name = 'correct_predict')
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")
with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    print('start assign \n')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1/moving_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1/moving_variance']))
    print('\n\nbn1 var assign finished ')
    # sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1/gamma']))
    # print('\n\nbn1 gamma assign finished ')

    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1/beta']))
    print('\n\nbn1 beta  assign finished ')

    for num in [[1, 2], [2, 2], [3, 2], [4, 2]]:
        for num_repeat in range(num[1]):
            name_scope = 'layer' + str(num[0]) + '.' + str(num_repeat) + '.bn1'
            tf_name_scope = name_scope + '/'
            for name in [ ['beta', 'beta:0'], ['moving_mean', 'moving_mean:0'],
                         ['moving_variance', 'moving_variance:0']]:
                sess.run(
                    tf.assign(graph.get_tensor_by_name(tf_name_scope + name[1]), value_dict[tf_name_scope + name[0]]))
                print(tf_name_scope + name[1])
            name_scope = 'layer' + str(num[0]) + '.' + str(num_repeat) + '.bn2'
            tf_name_scope = name_scope + '/'
            for name in [['beta', 'beta:0'], ['moving_mean', 'moving_mean:0'],
                         ['moving_variance', 'moving_variance:0']]:
                print(tf_name_scope + name[1])
                sess.run(
                    tf.assign(graph.get_tensor_by_name(tf_name_scope + name[1]), value_dict[tf_name_scope + name[0]]))

    for num in [2, 3, 4]:
        name_downsample_scope = 'layer' + str(num) + '.0.downsample.1'

        for name in [ ['beta', 'beta:0'], ['moving_mean', 'moving_mean:0'],
                     ['moving_variance', 'moving_variance:0']]:
            sess.run(tf.assign(graph.get_tensor_by_name(name_downsample_scope + '/' + name[1]),
                               value_dict[name_downsample_scope + '/' + name[0]]))
    print('\n\nAssign done\n\n')
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    print('coord done \n')

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars

    try:
        ckpt_root_path = './ckpt_stddev0.01'
        os.makedirs(ckpt_root_path, exist_ok=True)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        test_acc = 0
        test_count = 0
        print('\n\nstart validation\n\n')
        for i in range(int(np.floor(10000 / 256))):

            image_batch, label_batch = sess.run([test_images, test_labels])
            acc= sess.run(accuracy, feed_dict={x: image_batch, y: label_batch , is_training:False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("\n\nTest Accuracy before finetune: \n " + str(test_acc))

        num_lr = 1e-3

        for j in range(eval(FLAGS.epoch)):
            print('j=', j)
            print('\n\nstart training')
            if j == 0:
                print('initial learning_rate: ', num_lr, '\n')
            elif j % 4 == 0:
                num_lr = num_lr / 10
                print('learning_rate decay:', num_lr, '\n\n')

            # n = round(1281167 * eval(FLAGS.train_set_number_rate))  # 训练图片数量
            for i in range(int(np.floor(50000 / 256))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''is_trianing 必须设为False, 不然bn的mean 和variance会变'''
                _, loss_eval = sess.run([train_op, loss],feed_dict={x: image_batch, y: label_batch, is_training: False, lr: num_lr})
                if i == 0 and j == 0:
                    first_loss = loss_eval
                if i % 90 == 0:
                    print('j=' + str(j) + ' i=' + str(i) + '  loss=' + str(loss_eval))

                # test if bn layer change
                if i == 194:
                    var = 'layer2.1.bn1/moving_variance'
                    mean = 'layer2.1.bn1/moving_mean'
                    # gamma = 'layer2.1.bn1/gamma'
                    beta = 'layer2.1.bn1/beta'
                    print('\n\ntest if bn layer change')
                    print(np.sum(abs(value_dict[var] - sess.run(graph.get_tensor_by_name(var + ':0')))))
                    print(np.sum(abs(value_dict[mean] - sess.run(graph.get_tensor_by_name(mean + ':0')))))
                    # print(np.sum(abs(value_dict[gamma] - sess.run(graph.get_tensor_by_name(gamma + ':0')))))
                    print(np.sum(abs(value_dict[beta] - sess.run(graph.get_tensor_by_name(beta + ':0')))))
                    print('\n\n')

            test_acc = 0
            test_count = 0
            print('\nstart validation ,j=', j)
            for i in range(int(np.floor(10000 / 256))):
                image_batch, label_batch = sess.run([test_images, test_labels])
                acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("Test Accuracy:  " + str(test_acc))

        # saver.save(sess, ckpt_root_path + '/joint_com', global_step=j)
    finally:
        coord.request_stop()
        coord.join(threads)


