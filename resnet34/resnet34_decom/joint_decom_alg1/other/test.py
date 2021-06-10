


import tensorflow  as tf
import os
#ONLY CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# with tf.variable_scope('shape', reuse=tf.AUTO_REUSE):
#     a = tf.get_variable(name='a',  shape=[1], dtype=tf.float32,initializer=tf.constant_initializer(9))
#     b = tf.get_variable(name='b',  shape=[1], dtype=tf.float32,initializer=tf.constant_initializer(9))
# #
# x = tf.placeholder(tf.float32, [1], name ='x')
# y = tf.placeholder(tf.float32, [1], name='y')
#
# y_out = b *tf.nn.relu(a*x)
#
# loss = (y_out - y)*(y_out-y)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001, name='GradientDescent')
# train_op = optimizer.minimize(loss, var_list = [b], name='train_op')
#
# x_batch = [i for i in range(20)]
# y_ba = [i*4 for i in range(20)]
# with tf.Session() as sess:
#     sess.run( tf.global_variables_initializer() )
#     print(sess.run(b))
#     print(sess.run(a))
#     with tf.device('/cpu:0'):
#         for i in range(20):
#             print(i)
#             sess.run(train_op, feed_dict={x:[x_batch[i]], y:[y_ba[i]]})
#     print(sess.run(b))
#     print(sess.run(a))
#
#     saver = tf.train.Saver(max_to_keep=20)
#     saver.save(sess, './train_op_test', write_meta_graph=True)

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('C:/Users/Administrator/Desktop/csw/resnet34/resnet34_decom/joint_decom/train_op_test.meta')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    a  = graph.get_tensor_by_name("shape/a:0")
    b = graph.get_tensor_by_name("shape/b:0")
    train_op = graph.get_operation_by_name('train_op')

    sess.run(tf.global_variables_initializer())
    #这句话必须在initia后面，否则准确率就几乎为0
    saver.restore(sess, tf.train.latest_checkpoint('C:/Users/Administrator/Desktop/csw/resnet34/resnet34_decom/joint_decom'))
    # lr = tf.placeholder(tf.float32, name='learning_rate')
    # lr = graph.get_tensor_by_name('learning_rate:0')

    # is_training = graph.get_tensor_by_name('is_training:0')
    # accuracy = graph.get_tensor_by_name("accuracy:0")

    # train_op = graph.get_tensor_by_name(('train_op:0'))
    # loss = graph.get_tensor_by_name('loss:0')




    x_batch = [i for i in range(20)]
    y_ba = [i*4 for i in range(20)]

    print(sess.run(a))
    print(sess.run(b))

    for i in range(20):
        print(i)
        sess.run(train_op, feed_dict={x:[x_batch[i]], y:[y_ba[i]]})
    print(sess.run(a))
    print(sess.run(b))
