import tensorflow as tf
import numpy as np
import imagenet_data
import image_processing
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'



imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train')
val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=512, num_preprocess_threads=16)
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)







with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))



    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('/home/test01/csw/resnet34/for_meta/test.meta')

    saver.restore(sess, '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5')


    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    learning_rate = graph.get_tensor_by_name('learning_rate:0')

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

    saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

    test_acc = 0
    test_count = 0
    print('start vali first')
    for i in range(int(np.floor(50000 / 512))):
        image_batch, label_batch = sess.run([val_images, val_labels])
        acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("Test Accuracy:  " + str(test_acc))



    try:

        num_lr = 1e-3
        for j in range(10):

            print('\nstart training')
            if j%2==0 and j!=0:
                num_lr = num_lr / 5
                print('learning_rate decay:', num_lr)
            print('j=',j)

            for i in range(int(np.floor(1281167/128))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: True, learning_rate:num_lr})
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
            saver.save(sess, './ckpt3_20200617/rerun_from_5_acc0.7109375/acc_'+str(test_acc), global_step=j)



    finally:

        coord.request_stop()
        coord.join(threads)





