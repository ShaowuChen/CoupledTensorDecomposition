
import tensorflow as tf
import numpy as np
import os

flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('gpu', '1', 'gpu choosed to used' )

if FLAGS.gpu!='999':
    os.environ['CUDA_VISIBLE_DEVICES']= FLAGS.gpu
else:
    raise Exception('please name a gpu')

import sys

sys.path.append("/home/test01/csw/resnet34")
import imagenet_data

import image_processing
import math


imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train')
val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=512, num_preprocess_threads=16)
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)


def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num, is_training, flag_shortcut, down_sample, res_in_tensor=None):

    print('lay_num, repeat_num:', layer_num, ' ', repeat_num)
    print('shape of in_tesnor', in_tensor.shape)
    print('\n')

    if layer_num != 1 and repeat_num == 0 and conv_num == 1:
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]



    # 非共享
    if layer_num==1 or repeat_num==0:
        name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'  + 'conv' + str(conv_num) + '.weight'
        weight = tf.Variable(value_dict[name_weight], name=name_weight)

    else:#共享
        name_weight = 'layer' + str(layer_num) + '.'  + 'conv' + str(conv_num) + '.weight_ave_reuse'
        # if repeat_num==1: #首次，创建
        #     weight = tf.Variable(value_dict[name_weight], name=name_weight, dtype = tf.float32)
        # else: #reuse
        weight = tf.get_variable(name = name_weight, dtype = tf.float32, initializer=tf.constant(value_dict[name_weight ]))
        if weight not in var_list_to_train:
            var_list_to_train.append(weight )

    tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')

    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)
    # BN和Relu、shorcut照旧
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True, scale=True, epsilon=1e-9,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  is_training=is_training, scope=name_bn_scope)

    print('shape of tensor_go_next', tensor_go_next.shape)
    print('\n\n')
    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'

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


def basic_block(in_tensor, value_dict, layer_num, repeat_num, down_sample, is_training):



    tensor_go_next =  basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,
                                 is_training=is_training,flag_shortcut=False, down_sample=False,  res_in_tensor= None)
    tensor_go_next =  basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,
                                 is_training=is_training, flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor)
    return tensor_go_next



def repeat_basic_block(in_tensor, value_dict, repeat_times, layer_num, is_training):

    tensor_go_next = in_tensor
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for i in range(repeat_times):
            print(i,'start')
            down_sample = True if i==0 and layer_num!=1 else False
            tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training)
            print(i)
    return tensor_go_next





def resnet34(x, value_dict, is_training):

    weight_layer1 = tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,[1,2,2,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 1, is_training)
    print('\nblock1 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 4, 2, is_training)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 6, 3, is_training)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 4, is_training)
    print('\nblock4 finished')

    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=7, strides=1)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten, weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output







var_list_to_train = [] #保存共享训练参数

# value_dict = np.load('./joint_ave.npy', allow_pickle=True).item()

value_dict = np.load('./joint_ave_with_relu.npy', allow_pickle=True).item()

lr = tf.placeholder(tf.float32, name='learning_rate')
is_training = tf.placeholder(tf.bool, name = 'is_training')
x = tf.placeholder(tf.float32, [None,224,224,3], name = 'x')
y = tf.placeholder(tf.int64, [None], name="y")

print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training)
print('building network done\n\n')

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )
train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)

print('\n')
print('var_list_to_train')
print(var_list_to_train)
print('\n')
print('\n')



correct_predict = tf.equal(tf.argmax(output, 1), y, name = 'correct_predict')
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))


    print('start assign')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1/moving_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1/moving_variance']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1/gamma']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1/beta']))

    print('start assign2')
    for num in [[1,3],[2,4],[3,6],[4,3]]:
        for num_repeat in range(num[1]):
            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn1'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))

            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn2'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))

    print('start assign3')
    for num in [2,3,4]:
        name_downsample_scope = 'layer'+str(num)+'.0.downsample.1'

        for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
            sess.run(tf.assign(graph.get_tensor_by_name(name_downsample_scope +'/'+ name[1]), value_dict[name_downsample_scope +'/'+ name[0]]))

    print('assign done')

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars



    test_acc = 0
    test_count = 0
    print('\nstart first validation ')
    for i in range(int(np.floor(50000 / 512))):
        image_batch, label_batch = sess.run([val_images, val_labels])
        acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("first Test Accuracy:  " + str(test_acc)+'\n\n')


    '''raw save'''
    ckpt_root_path = './ckpt_ave_with_relu'
    log_path = ckpt_root_path + '/log.log'
    os.makedirs(ckpt_root_path + '/unfinetune_' + str(test_acc)[0:7], exist_ok=True)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
    saver.save(sess, ckpt_root_path + '/unfinetune_' + str(test_acc)[0:7] + '/ave_unfinetune_acc'+str(test_acc)[0:7], write_meta_graph=True)


    num_lr = 1e-2

    with open(log_path, 'a') as f:
        f.write("first Test Accuracy:  " + str(test_acc)+'\n')
        f.write("first learning_rate:  " + str(num_lr)+'\n\n\n\n\n')

    print('\nstart training')
    print('learning_rate start: ', num_lr,'\n\n')

    epoch = 20
    for j in range(epoch):
        print('\n\nj=', j)

        if num_lr == 1e-2:
            if j%2==0 and j!=0:
                num_lr = num_lr / 10
                print('learning_rate decay:', num_lr)

        else:
            if j%2==0 and j!=0 :
                num_lr = num_lr / 5
                print('learning_rate decay:', num_lr)


        for i in range(int(np.floor(1281167/128))):

            image_batch, label_batch = sess.run([train_images, train_labels])

            '''It is very important to set is_training 'False’， otherwise the moving_variance and moving_mean would changed'''
            _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: False, lr:num_lr})
            if i==0:
                start_loss = loss_eval
            if i%200 ==0:
                print('j='+str(j)+' i='+str(i)+ '  loss='+str(loss_eval))

            # test if bn layer change
            if j==0 and i == 300:
                var = 'layer3.2.bn1/moving_variance'
                mean = 'layer3.2.bn1/moving_mean'
                gamma = 'layer3.2.bn1/gamma'
                beta = 'layer3.2.bn1/beta'
                print('\n\ntest if bn layer change')
                print(np.sum(sess.run(graph.get_tensor_by_name(var + ':0'))))
                print(np.sum(sess.run(graph.get_tensor_by_name(mean + ':0'))))
                print(np.sum(abs(value_dict[var] - sess.run(graph.get_tensor_by_name(var + ':0')))))
                print(np.sum(abs(value_dict[mean] - sess.run(graph.get_tensor_by_name(mean + ':0')))))
                print(np.sum(abs(value_dict[gamma] - sess.run(graph.get_tensor_by_name(gamma + ':0')))))
                print(np.sum(abs(value_dict[beta] - sess.run(graph.get_tensor_by_name(beta + ':0')))))
                print('\n\n')



        test_acc = 0
        test_count = 0
        print('\nstart validation ,j=', j)
        for i in range(int(np.floor(50000 / 512))):
            image_batch, label_batch = sess.run([val_images, val_labels])
            acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Test Accuracy:  " + str(test_acc))

        with open(log_path, 'a') as f:
            f.write("\n\nj="+str(j) + "\nstart_loss= " + str(start_loss) + '\n')
            f.write("final_loss=  " + str(loss_eval) + '\n')
            f.write("test acc=  " + str(test_acc) + '\n')


        if j==0:
            saver.save(sess, ckpt_root_path+'/'+str(j)+'_acc'+str(test_acc)[0:7], global_step=j, write_meta_graph=True)
        else:
            saver.save(sess, ckpt_root_path+'/'+str(j)+'_acc'+str(test_acc)[0:7], global_step=j,write_meta_graph=False)



