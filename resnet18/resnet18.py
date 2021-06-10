import tensorflow as tf
import numpy as np
import os
from cifar10 import cifar10_input


def basic_conv(in_tensor, in_channels, out_channels, stride,  layer_num, repeat_num, conv_num, is_training, res_in_tensor=None):

    name_scope = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.'
    name_weight = name_scope + 'conv' + str(conv_num) + '.weight'
    name_bn_scope = name_scope + 'bn' + str(conv_num)

    if repeat_num==0 and conv_num==1:
        weight = tf.get_variable(shape=[3, 3, in_channels, out_channels], name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
    else:
        weight = tf.get_variable(shape=[3, 3, out_channels, out_channels], name=name_weight, dtype=tf.float32,   initializer=initializer, regularizer=regularizer)

    # weight = tf.get_variable(shape=[3,3,in_channels, out_channels], name=name_weight, dtype=tf.float32, initializer=initializer,regularizer=regularizer)

    print('\n'+name_scope+'conv'+str(conv_num))
    print(in_tensor.shape)

    strides = [1,stride, stride,1]
    tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, is_training=is_training, scope=name_bn_scope, )

    if conv_num==2:
        assert(res_in_tensor != None)
        if layer_num!=1 and repeat_num==0:
            name_downsample_weight = name_scope + 'downsample.0.weight'
            name_downsample_bn_scope = name_scope + 'downsample.1'

            weight_downsample = tf.get_variable(shape=[1, 1, in_channels, out_channels], name=name_downsample_weight,  dtype=tf.float32, initializer=initializer,regularizer=regularizer)
            shortcut = tf.nn.conv2d(res_in_tensor, weight_downsample, strides=[1, 2, 2, 1], padding='SAME')
            shortcut = tf.contrib.layers.batch_norm(shortcut,  scope=name_downsample_bn_scope, is_training=is_training)

        else:
            shortcut = res_in_tensor

        tensor_go_next = tensor_go_next + shortcut

    tensor_go_next = tf.nn.relu(tensor_go_next)
    print('\n' + name_scope + 'conv' + str(conv_num))
    print(tensor_go_next.shape)
    return tensor_go_next

def basic_block(in_tensor, in_channels, out_channels, layer_num, repeat_num, is_training):
    stride = 1
    if layer_num!=1 and repeat_num==0:
        stride=2
    tensor_go_next = basic_conv(in_tensor, in_channels, out_channels, stride, layer_num, repeat_num, 1, is_training)
    tensor_go_next = basic_conv(tensor_go_next, in_channels, out_channels, 1, layer_num, repeat_num, 2, is_training, res_in_tensor=in_tensor)
    return tensor_go_next

def repeat_basic_block(in_tensor, in_channels, out_channels, layer_num, repeat_times,is_training):
    tensor_go_next = in_tensor
    for i in range(repeat_times):
        print('\n\n'+str(i))
        tensor_go_next = basic_block(tensor_go_next, in_channels, out_channels, layer_num, i, is_training)
    return tensor_go_next

def resnet18(in_tensor, is_training):

    weight_layer1 = tf.get_variable(shape=[3, 3, 3, 64], name='conv1.weight', dtype=tf.float32,   initializer=initializer,regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight_layer1,strides=[1,1,1,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, is_training=is_training, scope='bn1')

    tensor_go_next = tf.nn.relu(tensor_go_next)

    #resnet18没有maxpool
    # tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    tensor_go_next = repeat_basic_block(tensor_go_next, 64,  64, layer_num=1, repeat_times=2, is_training=is_training)
    print('\nblock1 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 64, 128, layer_num=2, repeat_times=2, is_training=is_training)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 128, 256, layer_num=3, repeat_times=2,  is_training=is_training)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 256, 512, layer_num=4, repeat_times=2, is_training=is_training)  #(4,4,512)
    print('\nblock4 finished')

    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=4, strides=1) #(b,1,1,512)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten') #[b,512]
    weight_fc =  tf.get_variable(shape=[512, 10],  dtype=tf.float32, name='fc.weight',  initializer=initializer,regularizer=regularizer)
    bias_fc = tf.get_variable(shape=[10],  dtype=tf.float32, name='fc.bias', initializer=initializer,regularizer=regularizer)

    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten,weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output

flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('epoch', '240', 'train epoch' )

flags.DEFINE_string('gpu', '7', 'gpu choosed to used' )




os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.device('/cpu:0'):
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir='./cifar10/cifar-10-batches-bin', batch_size= 256)
    test_images, test_labels = cifar10_input.inputs(data_dir='./cifar10/cifar-10-batches-bin', eval_data=True, batch_size=256)


'''Network'''
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
is_training = tf.placeholder(tf.bool,name='is_training')
x =  tf.placeholder(tf.float32, [None,32,32,3],name='x')
y =  tf.placeholder(tf.int64, [None],name='y')
lr = tf.placeholder(tf.float32, name='learning_rate')

initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = tf.contrib.layers.l2_regularizer(5e-4) #L2 Regularizer

print('start building network')
logits, output = resnet18(x, is_training)
print('building network done\n\n')

keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss') + tf.add_n(keys)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9 , name='Momentum')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss,name='train_op')


correct_predict = tf.equal(tf.argmax(output, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")


with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

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

        ckpt_root_path = './ckpt_stddev0.01_test'
        os.makedirs(ckpt_root_path, exist_ok=True)
        log_path = ckpt_root_path+ '/log.log'

        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)


        '''Training'''
        num_lr = 0.1
        with open(log_path, 'a') as f:
            f.write('time:2020/6/20  Reset18 for Cifar10 \n\n')
            f.write('initial learning_rate: ' + str(num_lr))

        epoch = eval(FLAGS.epoch)
        print('\n\nstart training\n')
        for j in range(epoch):
            print('\n\nj= ', j)

            if j == 0:
                print('initial learning_rate: ', num_lr, '\n')
            if j==135:
                num_lr = 0.01
            elif j==185:
                num_lr = 0.001
                print('\nlearning_rate decay to:', num_lr, '\n')
            print('learning_rate:', num_lr)

            for i in range(int(np.ceil(50000 / 256))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch, is_training: True, lr: num_lr})
                if i == 0 and j == 0:
                    first_loss = loss_eval
                if i==0:
                    print('begin loss: ', loss_eval)
            print('end loss:', loss_eval)

            if j%5==0:
                test_acc = 0
                test_count = 0
                for i in range(int(np.floor(10000 / 256))):

                    image_batch, label_batch = sess.run([test_images, test_labels])
                    acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("Test Accuracy:  " + str(test_acc))

                with open(log_path, 'a') as f:
                    f.write('\n\nepoch= ' + str(j) + '\n')
                    f.write('learning_rate'+str(num_lr)+'\n')
                    if j == 0:
                        f.write('first loss: ' + str(first_loss) + '\n')
                    f.write('final loss: ' + str(loss_eval) + '\n')
                    f.write('accuracy_after_this_Train: ' + str(test_acc) + '\n\n')


            saver.save(sess, ckpt_root_path+'/resnet18-acc'+str(acc)[0:7], global_step=j)

    finally:
            coord.request_stop()
            coord.join(threads)






