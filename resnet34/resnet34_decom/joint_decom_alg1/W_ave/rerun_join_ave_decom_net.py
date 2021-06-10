import tensorflow as tf
import os

flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )
os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)

import sys

sys.path.append("/home/test01/csw/resnet34")



config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存

import numpy as np
import imagenet_data
import image_processing


imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train')
val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=512, num_preprocess_threads=16)
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)



with tf.Session(config=config) as sess:

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

    sess.run(tf.global_variables_initializer())


    saver = tf.train.import_meta_graph('./ckpt_ave/0_acc0.5936498397435898-0.meta')

    #这句话必须在initia后面，否则准确率就几乎为0
    saver.restore(sess, '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/W_ave/ckpt_ave/rerun/trained_acc0.6493589743589744-8')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    lr = graph.get_tensor_by_name('learning_rate:0')
    is_training = graph.get_tensor_by_name('is_training:0')

    accuracy = graph.get_tensor_by_name("accuracy:0")
    loss = graph.get_tensor_by_name('loss:0')
    train_op = graph.get_operation_by_name('train_op')


    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars

    print('==============')


    os.makedirs('./ckpt_ave/rerun/', exist_ok=True)
    f =  open('./ckpt_ave/rerun/restore_join_ave_log.log', 'a')
    f.write('start acc: 0.6493589743589744  (rerun -8)\n')
    try:
        num_lr = 1e-3
        for j in range(10):
            print('\n\nstart training j='+str(j)+'\n')

            if j!=0 and j%2==0:
                num_lr = num_lr / 10
            print('learning_rate decay:', num_lr)
            print('\nj=', j)
            f.write('j='+str(j)+'\n'+'learning_rate decay:'+str(num_lr)+'\n')

            for i in range(int(np.floor(1281167 / 128))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''分解实验中的is_training 必须是False,才能保证各个bn层的参数不变，只fine_tune分解层'''
                _, loss_eval = sess.run([train_op, loss],   feed_dict={x: image_batch, y: label_batch - 1, is_training: False, lr: num_lr})
                if i==0:
                    f.write('start loss:' + str(loss_eval)+'\n')

                if i % 200 == 0:
                    print('j=' + str(j) + ' i=' + str(i) + '  loss=' + str(loss_eval))

            f.write('final loss:' + str(loss_eval)+'\n')

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

            f.write('trained acc:' + str(test_acc)+'\n\n\n')

            saver = tf.train.Saver(max_to_keep=20)

            saver.save(sess, './ckpt_ave/rerun/trained_acc' + str(test_acc), global_step=j+10, write_meta_graph=False)
    finally:
        coord.request_stop()
        coord.join(threads)
        f.close()











