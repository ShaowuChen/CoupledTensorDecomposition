import tensorflow as tf
import numpy as np
import imagenet_data
import image_processing
import os
os.environ['CUDA_VISIBLE_DEVICES']='6'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num,
             flag_shortcut=False, down_sample=False,  is_training=False, res_in_tensor=None):

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

    weight_layer1 = tf.get_variable(initializer=tf.constant(value_dict['conv1.weight']), dtype = tf.float32, name='conv1.weight', regularizer=regularizer)
    name_bn1_scope = 'bn1'
    tensor_go_next =  tf.nn.conv2d(x, weight_layer1,[1,2,2,1], padding='SAME')
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next,  center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    #补上relu
    tensor_go_next = tf.nn.relu(tensor_go_next)



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
    weight_fc = tf.get_variable(initializer=tf.constant(value_dict['fc.weight'].T), dtype=tf.float32, name='fc.weight',regularizer=regularizer)
    print(value_dict['fc.weight'].shape,'\n\n\n\n')
    bias_fc = tf.get_variable(initializer=tf.constant(value_dict['fc.bias']), dtype=tf.float32, name='fc.bias',regularizer=regularizer)
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten,weight_fc) +bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output



'''Data'''
imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train')
val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=128, num_preprocess_threads=4)

#测试distorted_input效果
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)
# train_images, train_labels =  image_processing.distorted_inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=4)

value_dict = np.load('./resnet34.npy', allow_pickle=True).item()


'''Network'''
is_training = tf.placeholder(tf.bool,name='is_training')
x =  tf.placeholder(tf.float32, [None,224,224,3],name='x')
y =  tf.placeholder(tf.int64, [None],name='y')
lr = tf.placeholder(tf.float32, name='learning_rate')

regularizer = None
# regularizer = tf.contrib.layers.l2_regularizer(5e-4)

print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training)
print('building network done\n\n')





# keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss') + tf.add_n(keys)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9 , name='Momentum')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss,name='train_op')


correct_predict = tf.equal(tf.argmax(output, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")


with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    '''assign parateres from Pytorch to the net'''

    print('start to assign \n')
    graph = tf.get_default_graph()
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1.running_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1.running_var']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1.weight']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1.bias']))


    for num in [[1,3],[2,4],[3,6],[4,3]]:
        for num_repeat in range(num[1]):
            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn1'
            tf_name_scope = name_scope + '/'
            for name in [['weight','gamma:0'], ['bias','beta:0'], ['running_mean', 'moving_mean:0'], ['running_var', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[name_scope+'.'+name[0]]))
                print(tf_name_scope+name[1])

            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn2'
            tf_name_scope = name_scope + '/'
            for name in [['weight','gamma:0'], ['bias','beta:0'], ['running_mean', 'moving_mean:0'], ['running_var', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[name_scope+'.'+name[0]]))
                print(tf_name_scope + name[1])

    for num in [2,3,4]:
        name_downsample_scope = 'layer'+str(num)+'.0.downsample.1'

        for name in [['weight','gamma:0'], ['bias','beta:0'], ['running_mean', 'moving_mean:0'], ['running_var', 'moving_variance:0']]:
            sess.run(tf.assign(graph.get_tensor_by_name(name_downsample_scope +'/'+ name[1]), value_dict[name_downsample_scope+'.'+name[0]]))
            print(name_downsample_scope + name[1])

    print('assign done\n\n')



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
        '''raw validation'''
        print('\nraw validation')
        test_acc = 0
        test_count = 0
        print('\n\nstart validation\n\n')
        for i in range(int(np.floor(50000 / 128))):
            if i % 500 == 0:
                print(i)

            image_batch, label_batch = sess.run([val_images, val_labels])
            acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("raw  Test Accuracy: \n " + str(test_acc))

        '''raw save'''
        ckpt_root_path = './ckpt_with_relu_input'
        log_path = ckpt_root_path+ '/log.log'
        os.makedirs(ckpt_root_path+'/raw_from_pythorch_'+str(test_acc)[0:7],  exist_ok=True)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        saver.save(sess, ckpt_root_path+'/raw_from_pythorch_'+str(test_acc)[0:7]+'/resnet34-Pytorch', write_meta_graph=True)


        '''Training'''

        num_lr = 1e-3
        with open(log_path, 'a') as f:
            f.write('time:2020/6/17   改变is_training使baseline提升，后续分解bn层参数不变\n\n')
            f.write('\nfirst:\naccuracy_WithOut_Train: ' + str(test_acc) + '\n\n')
            f.write('initial learning_rate: ' + str(num_lr))

        epoch = 20
        for j in range(epoch):
            print('j=', j)
            print('\n\nstart training')
            if j == 0:
                print('initial learning_rate: ', num_lr, '\n')
            elif j % 2 == 0:
                num_lr = num_lr / 10
                print('\nlearning_rate decay:', num_lr, '\n')

            for i in range(int(np.floor(1281167 / 128))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''此处is_trianing 必须设为True；但是分解实验必须设为False 不然bn的mean 和variance会变'''
                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: True, lr: num_lr})
                if i == 0 and j == 0:
                    first_loss = loss_eval
                if i % 200 == 0:
                    print('j=' + str(j) + ' i=' + str(i) + '  loss=' + str(loss_eval))

            test_acc = 0
            test_count = 0
            print('\nstart validation ,j=', j)
            for i in range(int(np.floor(50000 / 128))):

                image_batch, label_batch = sess.run([val_images, val_labels])
                acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
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

            saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
            saver.save(sess, ckpt_root_path+'/resnet-34-acc'+str(accuracy), global_step=j)

    finally:
            coord.request_stop()
            coord.join(threads)

