import tensorflow as tf
import numpy as np
import os



import sys
sys.path.append("/home/test01/csw/resnet34")
import imagenet_data
import image_processing
import math





flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '0.5', 'how many train_set used to train')
flags.DEFINE_string('epoch', '4', 'train epoch' )

flags.DEFINE_string('rank_rate_yellow', None, 'rank_rate of each block1 conv2')

flags.DEFINE_string('rank_rate_ave', None, """Flag of type integer""")
flags.DEFINE_string('rank_rate_single', None, """Flag of type integer""")
flags.DEFINE_string('whole_ave', 'False', 'deecompose W_ave or not')
flags.DEFINE_string('ckpt_path', '20200720_120m_reload', 'rank_rate of each block1 conv2')
flags.DEFINE_string('iter', '0', 'iter num')
flags.DEFINE_string('gpu', '7', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate' )



ckpt_path = './'+FLAGS.ckpt_path+'/rankrate_yellow'+FLAGS.rank_rate_yellow[0]+'_'+FLAGS.rank_rate_yellow[2]+'_rankrate_ave'+FLAGS.rank_rate_ave[0]+'_'+FLAGS.rank_rate_ave[2]+\
            '/iter' +FLAGS.iter


if FLAGS.gpu=='all':
    pass
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)





#数据导入准备

imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train', eval(FLAGS.train_set_number_rate))

val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=512, num_preprocess_threads=16)
#256 is too large for linux to train---OOM
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)






with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))


    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/iter_way2/20200624_low_lr/rankrate_yellow3_4_rankrate_ave3_8/iter10/ckpt-3.meta')

    saver.restore(sess, '/home/test01/csw/resnet34/resnet34_decom/joint_decom_alg1/iter_way2/20200624_low_lr/rankrate_yellow3_4_rankrate_ave3_8/iter10/ckpt-3')





    graph = tf.get_default_graph()


    lr = graph.get_tensor_by_name('learning_rate:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name("y:0")

    loss = graph.get_tensor_by_name('loss:0')
    train_op = graph.get_operation_by_name("train_op")
    accuracy = graph.get_tensor_by_name("accuracy:0")




    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars



    saver = tf.train.Saver(var_list=var_list, max_to_keep=2)
    os.makedirs(ckpt_path,exist_ok=True)



    test_acc = 0
    test_count = 0
    print('\n\nstart validation\n\n')
    for i in range(int(np.floor(50000 / 512))):
        image_batch, label_batch = sess.run([val_images, val_labels])
        acc= sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training:False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("\nTest Accuracy: \n\n " + str(test_acc))

    # num_lr = 1e-1
    num_lr = eval(FLAGS.num_lr)
    with open(ckpt_path + '/log.log', 'a') as f:
        f.write('\nrank_rate_yellow: ' + FLAGS.rank_rate_yellow +'    rank_rate_ave: ' + FLAGS.rank_rate_ave + '\n')
        f.write('\naccuracy_WithOut_Train: ' + str(test_acc) + '\n\n')
        f.write('initial learning_rate: ' + str(num_lr))
        f.write('\nhow much train set to be used: ' + FLAGS.train_set_number_rate + '\n')




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


        with open(ckpt_path+'/log.log', 'a') as f:
            f.write('\n\nepoch= '+ str(j)+'\n')
            f.write('\n\nlearning_rate= ' + str(num_lr) + '\n')
            if j==0:

                f.write('first loss: ' + str(first_loss)+'\n')
            f.write('final loss: ' + str(loss_eval)+'\n')
            f.write('accuracy_after_this_Train: ' + str(test_acc)+'\n')




        if j!=0:
            saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
            saver.save(sess, ckpt_path+'/ckpt', global_step=j, write_meta_graph=False)
















    try:
        os.makedirs(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2]), exist_ok=True)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)


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




