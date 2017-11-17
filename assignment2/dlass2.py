import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def main(_):
    # hyperparams and configuration
    batch_size = 128
    learning_rate = 0.1
    epochs = 20
    logs_path = "/tmp/mnist/2"

    n_input = 784
    n_hidden_1 = 256
    n_hidden_2 = 256
    n_classes = 10

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # weights
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    W = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

    # biases
    b1 = tf.Variable(tf.zeros([n_hidden_1]))
    b2 = tf.Variable(tf.zeros([n_hidden_2]))
    b = tf.Variable(tf.zeros([n_classes]))

    # construct two layers with output y_logits
    layer_1 = tf.add(tf.matmul(X, W1), b1)
    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    y_logits = tf.nn.softmax(tf.matmul(layer_2, W) + b)

    # loss and optimizer
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_logits  + 1e-10), reduction_indices=[1]))
    # loss = tf.reduce_sum(
    #   tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_logits))

    opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(y_logits, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.scalar_summary('cross_entropy', loss)
    merged = tf.merge_all_summaries()



    losses = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
            
        # perform training cycles
        for epoch in range(epochs):
            num_batches = int(mnist.train.num_examples/batch_size)
            
            for i in range(num_batches):
                x_train, y_train = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, summary = sess.run([opt, merged], feed_dict={X: x_train, Y: y_train})
                writer.add_summary(summary, epoch * num_batches + i)

                
            print "Epoch: ", epoch 
        print "Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})



        print "done"


if __name__ == '__main__':
    tf.app.run(main=main)





