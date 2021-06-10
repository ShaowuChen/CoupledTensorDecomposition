import tensorflow as tf
import numpy as np
# import matplotlib.image as mpi
import os
import cifar10_input
# import matplotlib.pyplot as plt
print(tf.__version__)




classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')



# image = mpi.imread('cat.jpg')
# image = image/255
# image = (image- [0.4914, 0.4822, 0.4465]) / [0.2023, 0.1994, 0.2010]


os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



'''Network'''
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')



with tf.device('/cpu:0'):
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir='/home/test01/csw/resnet18/cifar-10-batches-bin', batch_size= 256)
    test_images, test_labels = cifar10_input.inputs(data_dir='/home/test01/csw/resnet18/cifar-10-batches-bin', eval_data=True, batch_size=250)







with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    print('coord done \n')

    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('/home/test01/csw/resnet10/ckpt_stddev0.01_test/resnet18-acc0.9375-230.meta')
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    is_training = graph.get_tensor_by_name("is_training:0")
    # output = graph.get_tensor_by_name('softmax:0')
    accuracy =  graph.get_tensor_by_name('accuracy:0')
    saver.restore(sess, '/home/test01/csw/resnet10/ckpt_stddev0.01_test/resnet18-acc0.9375-230')
    # output = sess.run(output, feed_dict={x:image, is_training:False})

    # print(output)
    # print(classes[np.argmax(output,axis=1)])



    test_acc = 0
    test_count = 0
    for i in range(int(np.floor(10000 / 250))):

        image_batch, label_batch = sess.run([test_images, test_labels])
        acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("Test Accuracy:  " + str(test_acc))


    test_acc = 0
    test_count = 0
    for i in range(int(np.floor(10000 / 250))):

        image_batch, label_batch = sess.run([test_images, test_labels])
        acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("Test Accuracy:  " + str(test_acc))







