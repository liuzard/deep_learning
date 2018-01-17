import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 训练参数
learning_rate=0.001
num_epochs=15
batch_size=100
display_step=10
example_to_show=10

#网络参数
num_input=784
num_hidden=200
num_construct=num_input

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)

def WeightsVariable(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out],stddev=0.1),dtype=tf.float32,name=name_str)

def BiasesVariable(n_bias,name_str):
    return tf.Variable(tf.zeros([n_bias]),dtype=tf.float32,name=name_str)

def encoder(x_origin,act_fun=tf.nn.sigmoid):
    with tf.name_scope('layer'):
        weights=WeightsVariable(num_input,num_hidden,'enc_weights')
        biases=BiasesVariable(num_hidden,'enc_biases')
        x_hidden=act_fun(tf.matmul(x_origin,weights)+biases)
        return x_hidden

def decoder(x_hidden,act_fun=tf.nn.sigmoid):
    with tf.name_scope('layer'):
        weights=WeightsVariable(num_hidden,num_construct,'dec_weights')
        biases=BiasesVariable(num_construct,'dec_biases')
        x_construct=act_fun(tf.matmul(x_hidden,weights)+biases)
        return x_construct

with tf.Graph().as_default():
    with tf.name_scope('input'):
        x=tf.placeholder(dtype=tf.float32,shape=[None,num_input],name='input')
    with tf.name_scope('encoder'):
        x_coder=encoder(x)
    with tf.name_scope('decoder'):
        x_decoder=decoder(x_coder)
    with tf.name_scope('loss'):
        loss=tf.reduce_mean(tf.pow(x_decoder-x,2))
    with tf.name_scope('train'):
        train=tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_num=int(mnist.train.num_examples/batch_size)
        for i in range(num_epochs):
            for j in range(train_num):
                xs,ys=mnist.train.next_batch(batch_size)
                _,loss_value=sess.run([train,loss],feed_dict={x:xs})
            if i%display_step==0:
                print(i+1,'=',loss_value)
        print("train finish")

        reconstruction=sess.run([x_decoder],feed_dict={x:mnist.test.images[0:example_to_show]})



# 原始图片和重构图片可视化
#         f,a=plt.subplots(2,10,figsize=(10,2))
#         for i in range(example_to_show):
#             a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
#             a[1][i].imshow(np.reshape(reconstruction[0][i], (28, 28)))
#         f.show()
#         plt.savefig('example.png')