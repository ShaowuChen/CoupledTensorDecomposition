import tensorflow as tf
import numpy as np
import load_data
import os
from tensorflow.python import pywrap_tensorflow

config = tf.ConfigProto()
config.gpu_options.allow_growth = True




flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('epoch', '100', 'train epoch' )


flags.DEFINE_string('ckpt_path', '20200720_net2_2', 'rank_rate of each block1 conv2')
flags.DEFINE_string('gpu', '7', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate' )

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu






def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num,flag_shortcut=False, down_sample=False,  is_training=False, res_in_tensor=None):

    print('lay_num, repeat_num:', layer_num, ' ', repeat_num)
    print('\n\n')
    print('shape of in_tesnor', in_tensor.shape)
    name_scope = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.'
    name_weight = name_scope + 'conv' + str(conv_num) + '.weight'
    name_bn_scope = name_scope + 'bn' + str(conv_num)


    if layer_num!=1 and repeat_num==0 and conv_num==1:
            strides = [1,2,2,1]
    else:
        strides = [1,1,1,1]
    weight = tf.get_variable(initializer=tf.constant(value_dict[name_weight]),dtype=tf.float32, name = name_weight,regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight,strides, padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next,  center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                             is_training=is_training, scope=name_bn_scope)



    if flag_shortcut==True:
        if down_sample==True:

            name_downsample_weight = name_scope + 'downsample.0.weight'
            name_downsample_bn_scope = name_scope + 'downsample.1'
            weight_down_sample = tf.Variable(value_dict[name_downsample_weight], name = name_downsample_weight)
            shorcut_tensor = tf.nn.conv2d(res_in_tensor, weight_down_sample,[1,2,2,1], padding='SAME')
            shorcut_tensor = tf.contrib.layers.batch_norm(shorcut_tensor,  center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_downsample_bn_scope)
        else:
            shorcut_tensor = res_in_tensor
        tensor_go_next = tensor_go_next + shorcut_tensor

    tensor_go_next = tf.nn.relu(tensor_go_next)
    return tensor_go_next



def basic_block(in_tensor, value_dict, layer_num, repeat_num, down_sample=False, is_training=False):

    tensor_go_next =  basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,
                                 flag_shortcut=False, down_sample=False,  res_in_tensor= None, is_training=is_training)

    tensor_go_next =  basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,
                                 flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor, is_training=is_training)
    return tensor_go_next


def repeat_basic_block(in_tensor, value_dict, repeat_times, layer_num, is_training):

    tensor_go_next = in_tensor
    for i in range(repeat_times):
        print(i,'start')
        down_sample = True if i==0 and layer_num!=1 else False
        tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training)
        print(i)
    return tensor_go_next



def resnet34(x, value_dict, is_training):

    'modify the first layer'

    weight_layer1 = tf.get_variable(shape=[3,3,3,64],initializer=tf.truncated_normal_initializer(stddev=0.01), dtype = tf.float32, name='conv1.weight', regularizer=regularizer)
    name_bn1_scope = 'bn1'
    tensor_go_next =  tf.nn.conv2d(x, weight_layer1,[1,1,1,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next,  center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)
    train_list.append(weight_layer1)

    tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 1, is_training)
    print('\nblock1 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 4, 2, is_training)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 6, 3, is_training)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 4, is_training)
    print('\nblock4 finished')

    'get rid of avg_pooling'
    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=2, strides=1)

    'retrain final fc layer'
    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.get_variable(shape=[512,100], initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=tf.float32, name='fc.weight',regularizer=regularizer)
    bias_fc = tf.get_variable(shape=[100],initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=tf.float32, name='fc.bias',regularizer=regularizer)
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten,weight_fc) +bias_fc

    train_list.append(weight_fc)
    train_list.append(bias_fc)


    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output


checkpoint_path = '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

value_dict = {}

for key in var_to_shape_map:
    value = reader.get_tensor(key)

    value_dict[key] = value



'''Network'''
is_training = tf.placeholder(tf.bool,name='is_training')
x =  tf.placeholder(tf.float32, [None,32,32,3],name='x')
y =  tf.placeholder(tf.int64, [None],name='y')
lr = tf.placeholder(tf.float32, name='learning_rate')

regularizer = None
# regularizer = tf.contrib.layers.l2_regularizer(5e-4)

train_list = []
print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training)
print('building network done\n\n')





# keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss') + tf.add_n(keys)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9 , name='Momentum')



update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op1 = optimizer.minimize(loss, name = 'train_op1',var_list = train_list) #只改变首末层参数（和bn层）

with tf.control_dependencies(update_ops):
    train_op2 = optimizer.minimize(loss, name = 'train_op2') #网络整体训练



correct_predict = tf.equal(tf.argmax(output, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")


with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    '''assign parateres from Pytorch to the net'''

    print('start to assign \n')
    graph = tf.get_default_graph()
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1/moving_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1/moving_variance']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1/gamma']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1/beta']))


    for num in [[1,3],[2,4],[3,6],[4,3]]:
        for num_repeat in range(num[1]):
            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn1'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))
                print(tf_name_scope + name[1])

            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn2'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                print(tf_name_scope+name[1])
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))

    for num in [2,3,4]:
        name_downsample_scope = 'layer'+str(num)+'.0.downsample.1'

        for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
            sess.run(tf.assign(graph.get_tensor_by_name(name_downsample_scope +'/'+ name[1]), value_dict[name_downsample_scope +'/'+ name[0]]))
    print('\n\nAssign done\n\n')


    test_img,test_labels = load_data.test()



    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars


    '''raw validation'''
    print('\nraw validation')
    test_acc = 0
    test_count = 0
    print('\n\nstart validation\n\n')
    test_img,test_labels = load_data.test()
    for i in range(int(np.floor(10000 / 250))):
        if i % 250 == 0:
            print(i)

        image_batch = test_img[i*250:(i+1)*250,:,:,:]
        label_batch = test_labels[i*250:(i+1)*250]
        acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("raw  Test Accuracy: \n " + str(test_acc))

    '''raw save'''
    ckpt_root_path = './ckpt_without_relu_input_all_layer_fine_tune'
    log_path = ckpt_root_path+ '/log.log'
    os.makedirs(ckpt_root_path+'/raw_from_pythorch_'+str(test_acc)[0:7],  exist_ok=True)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    saver.save(sess, ckpt_root_path+'/raw_from_pythorch_'+str(test_acc)[0:7]+'/resnet34-Pytorch', write_meta_graph=True)


    '''Training'''

    num_lr =  eval(FLAGS.num_lr)
    with open(log_path, 'a') as f:
        f.write('\nfirst:\naccuracy_WithOut_Train: ' + str(test_acc) + '\n\n')
        f.write('initial learning_rate: ' + str(num_lr))

    train_op = train_op1
    epoch = eval(FLAGS.epoch)
    for j in range(epoch):
        train_img, train_labels = load_data.train_shuffle()

        print('j=', j)
        print('\n\nstart training')

        if j==0:
            print('initial learning_rate: ', num_lr,'\n')
        elif (j+1)%100==0:                
            num_lr = num_lr / 10
            print('learning_rate: ', num_lr, '\n')

        #整体fine_tune
        if j==int(eval(FLAGS.epoch)/2)-1:
            train_op = train_op2
            # num_lr = 0.01
            num_lr = 0.1

       

        for i in range(int(np.floor(50000 / 250))):

            image_batch = train_img[i*250:(i+1)*250,:,:,:]
            label_batch = train_labels[i*250:(i+1)*250]

            _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch, is_training: True, lr: num_lr})
            if i == 0 and j == 0:
                first_loss = loss_eval
            if i % 200 == 0:
                print('j=' + str(j) + ' i=' + str(i) + '  loss=' + str(loss_eval))

        test_acc = 0
        test_count = 0
        print('\nstart validation ,j=', j)
        for i in range(int(np.floor(10000 / 250))):

            image_batch = test_img[i*250:(i+1)*250,:,:,:]
            label_batch = test_labels[i*250:(i+1)*250]
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

        if j!=0 and (j+1)%50==0:
            saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
            saver.save(sess, ckpt_root_path+'/resnet-34-acc'+str(test_acc), global_step=j,write_meta_graph=False)



