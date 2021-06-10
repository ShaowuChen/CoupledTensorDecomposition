import numpy as np
import os
import tensorflow as tf



flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '0.5', 'how many train_set used to train')
flags.DEFINE_string('epoch', '4', 'train epoch' )


flags.DEFINE_string('rank_rate', '1/2', """Flag of type integer""")
flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate')

flags.DEFINE_string('ckpt_path', '20200720_12m_from_reload', 'enough_epoch_to_check_structure' )
ckpt_path = './'+FLAGS.ckpt_path


if FLAGS.gpu=='all':
    pass
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存




import sys
sys.path.append("/home/test01/csw/resnet34")
import imagenet_data
import image_processing
import math



#数据导入准备

imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train', eval(FLAGS.train_set_number_rate))

val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=512, num_preprocess_threads=16)
#256 is too large for linux to train---OOM
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)



#获取G1 G2 G3
def decompose(value, rank):
    shape = value.shape
    h = shape[0]
    w = shape[1]
    i = shape[2]
    o = shape[3]
    assert (len(shape) == 4)
    assert (h == 3)
    assert (w == 3)
    assert (o >= i)

    rank1=rank2=rank

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



def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num, flag_shortcut, down_sample, is_training, res_in_tensor=None, rank_rate=None):
    print('lay_num, repeat_num:', layer_num, ' ', repeat_num)
    print('shape of in_tesnor', in_tensor.shape)
    print('\n')

    if layer_num != 1 and repeat_num == 0 and conv_num == 1:
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]

    # 64不分解；其余的第一个block的第一个conv
    if layer_num==1 or (repeat_num==0 and conv_num==1):
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
        rank = round(shape[2]*rank_rate)

        os.makedirs(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/', exist_ok=True)
        with open(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'_log.log', 'a') as f:
            f.write(name_scope + ' ratio=' + FLAGS.rank_rate + ' rank=' + str(rank) + '\n')

        G3,G2,G1 = decompose(value_dict[name_weight],rank)
        # 1层卷积变3层 (Weight [H,W,I,O])
        weight1 = tf.Variable(G1, name=name_weight1)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')

        weight2 = tf.Variable(G2, name=name_weight2)
        tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')

        weight3 = tf.Variable(G3, name=name_weight3)
        tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')

        #加入训练list;冻结其他一切层
        var_list_to_train.append(weight1)
        var_list_to_train.append(weight2)
        var_list_to_train.append(weight3)


    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)

    # BN和Relu、shorcut照旧
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True, scale=True, epsilon=1e-9,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  is_training=is_training, scope=name_bn_scope)

    print('shape of tensor_go_next', tensor_go_next.shape)
    print('\n\n')

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



def basic_block(in_tensor, value_dict, layer_num, repeat_num, down_sample, is_training=False, rank_rate=None):



    tensor_go_next =  basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,
                                     is_training=is_training,flag_shortcut=False, down_sample=False,  res_in_tensor= None, rank_rate=rank_rate)
    tensor_go_next =  basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,
                                     is_training=is_training, flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor, rank_rate=rank_rate)
    return tensor_go_next



def repeat_basic_block(in_tensor, value_dict, repeat_times, layer_num, is_training, rank_rate=None):

    tensor_go_next = in_tensor
    for i in range(repeat_times):
        down_sample = True if i == 0 and layer_num != 1 else False
        # if i==0:
        #     tensor_go_next = orig_basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training)
        # else:

        tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training, rank_rate=rank_rate)
    return tensor_go_next




def resnet34(x, value_dict, is_training, rank_rate):


    weight_layer1 = tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,[1,2,2,1], padding='SAME')

    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


    '''64 block1不分解'''
    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 1, is_training, rank_rate=None)
    print('\nblock1 finished')


    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 4, 2, is_training, rank_rate)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 6, 3, is_training, rank_rate)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 4, is_training, rank_rate)
    print('\nblock4 finished')


    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=7, strides=1)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten, weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output




var_list_to_train = []  #保存要训练的G1，G2，G3;冻结其他参数；
rank_rate = eval(FLAGS.rank_rate)

value_dict = np.load('./parameter.npy', allow_pickle=True).item()

lr = tf.placeholder(tf.float32, name='learning_rate')
is_training = tf.placeholder(tf.bool, name = 'is_training')
x = tf.placeholder(tf.float32, [None,224,224,3], name = 'x')
y = tf.placeholder(tf.int64, [None], name="y")


print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training, rank_rate)
print('building network done\n\n')





loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )
train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)

'''这个可能就是导致训练时 is_training=False 出现nan mean / var的罪魁祸首'''
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)


correct_predict = tf.equal(tf.argmax(output, 1), y, name = 'correct_predict')
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")



with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()



    print('start assign \n')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1/moving_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1/moving_variance']))
    print('\n\nbn1 var assign finished ')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1/gamma']))
    print('\n\nbn1 gamma assign finished ')

    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1/beta']))
    print('\n\nbn1 beta  assign finished ')


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
        os.makedirs(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/single_tt_raw/', exist_ok=True)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        saver.save(sess, ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/single_tt_raw/'+'single_tt_raw', write_meta_graph=True)



        test_acc = 0
        test_count = 0
        print('\n\nstart validation\n\n')
        for i in range(int(np.floor(50000 / 512))):
            if i%500==0:
                print(i)

            image_batch, label_batch = sess.run([val_images, val_labels])
            acc= sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training:False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("\n\nTest Accuracy: \n " + str(test_acc))

        num_lr = eval(FLAGS.num_lr)
        with open(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'/'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'_log.log', 'a') as f:
            f.write('\naccuracy_WithOut_Train: ' + str(test_acc)+'\n\n')
            f.write('initial learning_rate: ' + str(num_lr))
            f.write('\nhow much train set to be used: '+ FLAGS.train_set_number_rate+'\n')

        print('how much train set to be used: ', FLAGS.train_set_number_rate)


        for j in range(eval(FLAGS.epoch)):
            print('j=', j)
            print('\n\nstart training')
            if j==0:
                print('initial learning_rate: ', num_lr,'\n')


            elif j%2==0:
                num_lr = num_lr / 10 
                print('learning_rate: ', num_lr, '\n')


            n=round(1281167*eval(FLAGS.train_set_number_rate))  #训练图片数量
            for i in range(int(np.floor(n/128))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''is_trianing 必须设为False, 不然bn的mean 和variance会变'''
                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: False, lr:num_lr})
                if i==0 and j==0:
                    first_loss = loss_eval
                if i%200 ==0:
                    print('j='+str(j)+' i='+str(i)+ '  loss='+str(loss_eval))




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


            with open(ckpt_path+'/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '_log.log', 'a') as f:
                if j==0:
                    f.write('rank_rate:'+str(FLAGS.rank_rate)+'\n')
                    f.write('Num_Train_picture: '+str(n)+'\n')
                    f.write('first loss: ' + str(first_loss)+'\n')

                f.write('\nepoch= '+ str(j)+'\n')

                f.write('learning_rate: '+str(num_lr)+'\n')
                f.write('final loss: ' + str(loss_eval)+'\n')
                f.write('accuracy_after_this_Train: ' + str(test_acc)+'\n\n')


            saver.save(sess, ckpt_path+'/rank_rate' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2]) + '/' + str(FLAGS.rank_rate[0]) + '_' + str(FLAGS.rank_rate[2])+'_acc'+str(test_acc), global_step=j)

    finally:
            coord.request_stop()
            coord.join(threads)



