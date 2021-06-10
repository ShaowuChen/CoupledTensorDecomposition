
'''
修改日志：
2020/6/22 将 W_diff= W原 - W_ave 改为 W_diff = W原 - reverse(G_ave1,G_ave2,G_ave3), ckpt为ckpt_finetuned_ave_20200622_lr0.01


'''


import numpy as np
import os
import tensorflow as tf
import iter2_odd_svd
import iter2_odd_svd_2


flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '0.5', 'how many train_set used to train')
flags.DEFINE_string('epoch', '4', 'train epoch' )
flags.DEFINE_string('svd','0','svd')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")
flags.DEFINE_string('whole_ave', 'False', 'deecompose W_ave or not')
flags.DEFINE_string('ckpt_path', '20200729_fine_tune_60w', 'ckpt_path')
flags.DEFINE_string('iter', '0', 'iter num')
flags.DEFINE_string('gpu', '7', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate' )




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

from tensorflow.python import pywrap_tensorflow



#数据导入准备

imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train', eval(FLAGS.train_set_number_rate))

val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=500, num_preprocess_threads=4)
#256 is too large for linux to train---OOM
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=4)




# 获取G1 G2 G3
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



def basic_conv_ave(in_tensor, value_dict, layer_num, repeat_num,  conv_num, strides):

    assert(layer_num!=1)
    if repeat_num==0 and conv_num==1:
        name_weight_ave = 'layer' +str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight_ave_reuse'
        name_weight_ave_common = 'layer' + str(layer_num) + '.conv' +str(conv_num)+'.weight_ave_reuse'
        name_weight_ave1 = name_weight_ave + '1'
        name_weight_ave2 = name_weight_ave_common + '2'
        name_weight_ave3 = name_weight_ave_common + '3'
    else:
        name_weight_ave = 'layer' + str(layer_num) + '.conv' +str(conv_num)+'.weight_ave_reuse'
        name_weight_ave1 = name_weight_ave + '1'
        name_weight_ave2 = name_weight_ave + '2'
        name_weight_ave3 = name_weight_ave + '3'


    # 1层卷积变3层 (Weight [H,W,I,O])
    weight_ave1 = tf.get_variable(name=name_weight_ave1, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight_ave1]),regularizer=regularizer) #共享
    tensor_go_next = tf.nn.conv2d(in_tensor, weight_ave1, [1, 1, 1, 1], padding='SAME')

    weight_ave2 = tf.get_variable(name=name_weight_ave2, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight_ave2]),regularizer=regularizer) #共享
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight_ave2, [1, strides[1], strides[2], 1], padding='SAME')

    weight_ave3= tf.get_variable(name=name_weight_ave3, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight_ave3]),regularizer=regularizer) #共享
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight_ave3, [1, 1, 1, 1], padding='SAME')

    # 加入训练list;冻结其他一切层
    var_list_to_train.append(weight_ave1)
    var_list_to_train.append(weight_ave2)
    var_list_to_train.append(weight_ave3)

    return tensor_go_next



def basic_conv_single(in_tensor, value_dict, layer_num, repeat_num, conv_num,strides, rank_rate_single=None):

    orig_name = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.weight'
    diff_name = orig_name + '_diff'

    name_weight1 = diff_name + '1'
    name_weight2 = diff_name + '2'
    name_weight3 = diff_name + '3'

    # 1层卷积变3层 (Weight [H,W,I,O])
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight1]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')
    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, strides[1], strides[2], 1], padding='SAME')
    weight3 = tf.get_variable(name=name_weight3, dtype=tf.float32, initializer=tf.constant(value_dict[name_weight3]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')
    #加入训练list;冻结其他一切层
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)
    var_list_to_train.append(weight3)

    return tensor_go_next




def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num,
               flag_shortcut, down_sample, is_training, res_in_tensor, rank_rate_ave, rank_rate_single, whole_ave):

    if layer_num != 1 and (repeat_num == 0 and conv_num == 1):
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]

    # layer1不分解;
    if layer_num==1:
        name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'  + 'conv' + str(conv_num) + '.weight'
        weight = tf.Variable(value_dict[name_weight], name=name_weight)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')

    else:
        tensor_from_ave = basic_conv_ave(in_tensor, value_dict, layer_num, repeat_num, conv_num, strides=strides)
        tensor_from_single = basic_conv_single(in_tensor, value_dict, layer_num, repeat_num, conv_num, strides=strides, rank_rate_single=rank_rate_single)

        tensor_go_next = tensor_from_single + tensor_from_ave

    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)
    # BN和Relu、shorcut照旧
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True, scale=True, epsilon=1e-9,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  is_training=is_training, scope=name_bn_scope)
    # print('shape of tensor_go_next', tensor_go_next.shape)


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

def basic_block(in_tensor, value_dict, layer_num, repeat_num, down_sample=False, is_training=False, rank_rate_ave=None,rank_rate_single=None, whole_ave=None):


    tensor_go_next =  basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,
                                     is_training=is_training,flag_shortcut=False, down_sample=False,  res_in_tensor= None, rank_rate_ave = rank_rate_ave, rank_rate_single = rank_rate_single, whole_ave = whole_ave)
    tensor_go_next =  basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,
                                     is_training=is_training, flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor, rank_rate_ave = rank_rate_ave, rank_rate_single = rank_rate_single, whole_ave = whole_ave)
    return tensor_go_next

def repeat_basic_block(in_tensor, value_dict, repeat_times, layer_num, is_training, rank_rate_ave, rank_rate_single, whole_ave):

    tensor_go_next = in_tensor
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for i in range(repeat_times):
            down_sample = True if i == 0 and layer_num != 1 else False
            tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training, rank_rate_ave=rank_rate_ave,rank_rate_single=rank_rate_single, whole_ave=whole_ave)
    return tensor_go_next

def resnet34(x, value_dict, is_training,  rank_rate_ave, rank_rate_single, whole_ave):


    weight_layer1 = tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,[1,2,2,1], padding='SAME')

    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)

    tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


    '''64 block1不分解'''
    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 1, is_training, rank_rate_ave=None, rank_rate_single=None, whole_ave=None)
    print('\nblock1 finished')


    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 4, 2, is_training, rank_rate_ave, rank_rate_single, whole_ave)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 6, 3, is_training, rank_rate_ave, rank_rate_single, whole_ave)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 4, is_training, rank_rate_ave, rank_rate_single, whole_ave)
    print('\nblock4 finished')


    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=7, strides=1)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten, weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output




if FLAGS.rank_rate_ave is not None:
    rank_rate_ave = eval(FLAGS.rank_rate_ave)
else:
    rank_rate_ave = None
rank_rate_single = eval(FLAGS.rank_rate_single)
whole_ave = eval(FLAGS.whole_ave)



'''add high acc ave to train with L2 fail '''

if FLAGS.svd=='0':
    ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]+'_iter'+FLAGS.iter
elif FLAGS.svd=='2':
    ckpt_path = './'+FLAGS.ckpt_path+'_2'+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]+'_iter'+FLAGS.iter

# ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]+'_iter'+FLAGS.iter
os.makedirs(ckpt_path, exist_ok=True)

brief_log_path = './'+FLAGS.ckpt_path




value_path_npy = '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/iter_way2_and_decom_first_layer/20200704'+'/rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+'_rankrate_single'+FLAGS.rank_rate_single[0:]

value_dict_even = np.load(value_path_npy+'/iter_even'+FLAGS.iter+'.npy', allow_pickle=True).item()
# value_dict_odd = np.load(value_path_npy+'/iter_odd'+FLAGS.iter+'.npy', allow_pickle=True).item()

if FLAGS.svd=='0':
    value_dict_odd = iter2_odd_svd.iter_odd(rank_rate_ave, rank_rate_single, eval(FLAGS.iter))
elif FLAGS.svd=='2':
    value_dict_odd = iter2_odd_svd_2.iter_odd(rank_rate_ave, rank_rate_single, eval(FLAGS.iter))
# value_dict_odd = iter2_odd_svd_2.iter_odd(rank_rate_ave, rank_rate_single, eval(FLAGS.iter))
value_dict_even.update(value_dict_odd)
value_dict = value_dict_even


lr = tf.placeholder(tf.float32, name='learning_rate')
is_training = tf.placeholder(tf.bool, name = 'is_training')
x = tf.placeholder(tf.float32, [None,224,224,3], name = 'x')
y = tf.placeholder(tf.int64, [None], name="y")

# regularizer = tf.contrib.layers.l2_regularizer(5e-4)
regularizer = None
var_list_to_train = []  #保存要训练的G1，G2，G3;冻结其他参数；

print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training,  rank_rate_ave, rank_rate_single, whole_ave)
print('building network done\n\n')




#
# keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss') + tf.add_n(keys)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )
train_op = optimizer.minimize(loss, name='train_op', var_list=var_list_to_train)

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

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=var_list, max_to_keep=2)
    os.makedirs(ckpt_path+'/joint_raw',exist_ok=True)

    # saver.save(sess, ckpt_path+'/joint_raw/untrain_model', write_meta_graph=True)



    test_acc = 0
    test_count = 0
    print('\n\nstart validation\n\n')
    for i in range(int(np.floor(50000 / 500))):
        image_batch, label_batch = sess.run([val_images, val_labels])
        acc= sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training:False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("\nTest Accuracy: \n\n " + str(test_acc))

    # num_lr = 1e-1
    num_lr = eval(FLAGS.num_lr)
    with open(ckpt_path + '/log.log', 'a') as f:
        f.write('\nrank_rate_ave: ' + FLAGS.rank_rate_ave +'    rank_rate_single: ' + FLAGS.rank_rate_single + '\n')
        f.write('\naccuracy_WithOut_Train: ' + str(test_acc) + '\n\n')
        f.write('initial learning_rate: ' + str(num_lr))
        f.write('\nhow much train set to be used: ' + FLAGS.train_set_number_rate + '\n')

    with open(brief_log_path+'/log1.log', 'a') as f:
        write1 = '\nrank_rate_ave: ' + FLAGS.rank_rate_ave +'    rank_rate_single: ' + FLAGS.rank_rate_single + '\n'
        write2 = 'iter: '+FLAGS.iter+'\n'
        write3 = 'initial learning_rate: ' + str(num_lr) +'\n'
        write4 = 'accuracy_WithOut_Train: ' + str(test_acc) + '\n\n'
        f.write(write1 + write2 +write3+write4)


    print('how much train set to be used: ', FLAGS.train_set_number_rate)

    for j in range(eval(FLAGS.epoch)):
        print('j=', j)
        print('\n\nstart training')
        if j==0:
            print('initial learning_rate: ', num_lr,'\n')
        # else:
    
        else:
    
            num_lr = num_lr / 10
            print('learning_rate: ', num_lr, '\n')
            # print('learning_rate decay:', num_lr, '\n\n')
    
    
        n=round(1281167*eval(FLAGS.train_set_number_rate))  #训练图片数量
        for i in range(int(np.floor(n/128))):
    
            image_batch, label_batch = sess.run([train_images, train_labels])
    
            _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: False, lr:num_lr})
            if i==0 and j==0:
                first_loss = loss_eval
            if i%200 ==0:
                print('j='+str(j)+' i='+str(i)+ '  loss='+str(loss_eval))
    
    
        test_acc = 0
        test_count = 0
        print('\nstart validation ,j=', j)
        for i in range(int(np.floor(50000 / 500))):
    
            image_batch, label_batch = sess.run([val_images, val_labels])
            acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Test Accuracy:  " + str(test_acc))
    
    
        with open(ckpt_path+'/log.log', 'a') as f:
            f.write('\n\nepoch= '+ str(j)+'\n')
            f.write('\n\nlearning_rate= ' + str(num_lr) + '\n')
            if j==0:
    
                f.write('first loss: ' + str(first_loss)+'\n')
            f.write('final loss: ' + str(loss_eval)+'\n')
            f.write('accuracy_after_this_Train: ' + str(test_acc)+'\n')
    
        if j==3:
            with open(brief_log_path + '/log1.log', 'a') as f:
                write1 = '\nFine_tuned:\nrank_rate_ave: ' + FLAGS.rank_rate_ave + '    rank_rate_single: ' + FLAGS.rank_rate_single + '\n'
                write2 = 'iter: ' + FLAGS.iter + '\n'
                write3 = 'initial learning_rate: ' + str(num_lr) + '\n'
                write4 = 'epoch4 Fine_tuned accuracy: ' + str(test_acc) + '\n\n'
                f.write(write1 + write2 + write3 + write4)
    
        if j==3:
            saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
            saver.save(sess, ckpt_path+'/ckpt', global_step=j, write_meta_graph=False)




