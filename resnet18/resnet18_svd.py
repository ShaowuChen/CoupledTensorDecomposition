import tensorflow as tf
import numpy as np
import os
from cifar10 import cifar10_input
import math
import sys
flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '0.5', 'how many train_set used to train')
flags.DEFINE_string('epoch', '8', 'train epoch' )


flags.DEFINE_string('rank_rate', '0.31770833', """Flag of type integer""")
flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )
if FLAGS.gpu=='all':
    pass
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#获取G1 G2
def decompose(weight, rank_rate):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]
    rank=int(np.floor(O*rank_rate))
    W = np.reshape(np.transpose(weight,(0,2,1,3)), [F1*I, -1])
    u1, s1, v1 = np.linalg.svd(W, full_matrices=True)
    G1  = np.dot(u1[:, 0:rank].copy(),np.diag(s1)[0:rank, 0:rank].copy())
    G1 = np.transpose(np.reshape(G1, [F1, I, 1, rank]),(0,2,1,3))
    G2 = v1[0:rank, :].copy()
    G2=np.transpose(np.reshape(G2,[1,rank,F2,O]),(0,2,1,3))

    return G2,G1

def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num, flag_shortcut,down_sample, is_training, res_in_tensor=None, rank_rate=None):

    if layer_num != 1 and repeat_num == 0 and conv_num == 1:
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]

    # 64不分解
    if layer_num==1:
        name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'  + 'conv' + str(conv_num) + '.weight'
        weight = tf.Variable(value_dict[name_weight], name=name_weight)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')


    else:#分解
        name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
        name_weight = name_scope + 'conv' + str(conv_num) + '.weight'

        name_weight1 = name_weight + '1'
        name_weight2 = name_weight + '2'
        name_weight3 = name_weight + '3'

        shape = value_dict[name_weight].shape
        # rank = round(shape[2]*rank_rate)

        os.makedirs('./ckpt2/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/', exist_ok=True)
        with open('./ckpt2/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'_log.log', 'a') as f:
            f.write(name_scope + ' ratio=' + FLAGS.rank_rate + ' rank_rate=' + str(rank_rate) + '\n')

        G2,G1 = decompose(value_dict[name_weight],rank_rate)
        # 1层卷积变3层 (Weight [H,W,I,O])
        weight1 = tf.Variable(G1, name=name_weight1)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')

        weight2 = tf.Variable(G2, name=name_weight2)
        tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')


        #加入训练list;冻结其他一切层
        var_list_to_train.append(weight1)
        var_list_to_train.append(weight2)



    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)

    # BN和Relu、shorcut照旧
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True,  epsilon=1e-9,
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
            shorcut_tensor = res_in_tensor

        tensor_go_next = tensor_go_next + shorcut_tensor

    tensor_go_next = tf.nn.relu(tensor_go_next)
    return tensor_go_next

def basic_block(in_tensor, value_dict,  layer_num, repeat_num, down_sample, is_training=False, rank_rate=None):

    tensor_go_next = basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,is_training=is_training,
                                flag_shortcut=False, down_sample=False,  res_in_tensor= None, rank_rate=rank_rate)
    tensor_go_next = basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,is_training=is_training,
                                flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor, rank_rate=rank_rate)
    return tensor_go_next

def repeat_basic_block(in_tensor, value_dict, repeat_times,layer_num, is_training, rank_rate=None):
    tensor_go_next = in_tensor
    for i in range(repeat_times):
        down_sample = True if i == 0 and layer_num != 1 else False
        print('\n'+str(i))
        tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample,is_training=is_training, rank_rate=rank_rate)
    return tensor_go_next

def resnet18(x,value_dict,is_training, rank_rate):

    weight_layer1 =  tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,strides=[1,1,1,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next,  decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    #resnet34忘了加relu
    tensor_go_next = tf.nn.relu(tensor_go_next)
    '''64 layer1不分解'''
    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, repeat_times=2, layer_num=1, is_training=is_training, rank_rate=None)
    print('layer1 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,repeat_times=2, layer_num=2,  is_training=is_training, rank_rate=rank_rate)
    print('layer2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, repeat_times=2, layer_num=3,  is_training=is_training, rank_rate=rank_rate)
    print('layer3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict,  repeat_times=2,layer_num=4, is_training=is_training, rank_rate=rank_rate)  #(4,4,512)
    print('layer4 finished')

    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=4, strides=1) #(b,1,1,512)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten') #[b,512]
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')

    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten,weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output

var_list_to_train = []  #保存要训练的G1，G2，G3;冻结其他参数；
rank_rate = eval(FLAGS.rank_rate)

value_dict = np.load('./parameter.npy', allow_pickle=True).item()

with tf.device('/cpu:0'):
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir='../cifar10/cifar-10-batches-bin', batch_size= 256)
    test_images, test_labels = cifar10_input.inputs(data_dir='../cifar10/cifar-10-batches-bin', eval_data=True, batch_size=256)


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
tensor_go_next_fc, output = resnet18(x, value_dict, is_training, rank_rate)
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
        os.makedirs('./ckpt2/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/single_tt_raw/',exist_ok=True)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        saver.save(sess, './ckpt2/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/single_tt_raw/'+'single_tt_raw', write_meta_graph=True)
        test_acc = 0
        test_count = 0
        for i in range(int(np.floor(10000 / 256))):

            image_batch, label_batch = sess.run([test_images, test_labels])
            acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Test Accuracy before finetune:  " + str(test_acc))

        num_lr = 1e-3
        with open('./ckpt2/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/' + str(
                FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '_log.log', 'a') as f:
            f.write('\naccuracy_WithOut_Train: ' + str(test_acc) + '\n\n')
            f.write('initial learning_rate: ' + str(num_lr))
            # f.write('\nhow much train set to be used: ' + FLAGS.train_set_number_rate + '\n')

        # print('how much train set to be used: ', FLAGS.train_set_number_rate)

        for j in range(eval(FLAGS.epoch)):
            print('j=', j)
            print('\n\nstart training')
            if j == 0:
                print('initial learning_rate: ', num_lr, '\n')
            elif j % 2 == 0:
                num_lr = num_lr / 10
                print('learning_rate decay:', num_lr, '\n\n')

            # n = round(1281167 * eval(FLAGS.train_set_number_rate))  # 训练图片数量
            for i in range(int(np.floor(50000 / 256))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''is_trianing 必须设为False, 不然bn的mean 和variance会变'''
                _, loss_eval = sess.run([train_op, loss],
                                        feed_dict={x: image_batch, y: label_batch, is_training: False, lr: num_lr})
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

            with open('./ckpt2/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/' + str(
                    FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '_log.log', 'a') as f:
                f.write('\nepoch= ' + str(j) + '\n')
                if j == 0:
                    f.write('Num_Train_picture: ' + str(50000) + '\n')
                    f.write('first loss: ' + str(first_loss) + '\n')
                f.write('final loss: ' + str(loss_eval) + '\n')
                f.write('accuracy_after_this_Train: ' + str(test_acc) + '\n\n')

            saver.save(sess, './ckpt2/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/' + str(
                FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]), global_step=j)

    finally:
        coord.request_stop()
        coord.join(threads)
