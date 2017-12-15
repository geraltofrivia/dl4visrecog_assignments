import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np


def main(_):
    data = np.load('data/ORL_faces.npz')
    trainX = data['trainX']
    trainY = data['trainY'].astype(int)
    testX = data['testX']
    testY = data['testY'].astype(int)

    print testY.shape, trainY.shape

    temp = np.zeros(trainY.shape + (20,), dtype=np.int8)
    temp[np.arange(trainY.shape[0]), trainY] = 1
    trainY = temp

    temp = np.zeros(testY.shape + (20,), dtype=np.int8)
    temp[np.arange(testY.shape[0]), testY] = 1
    testY = temp

    # hyperparams and configuration
    batch_size = 10
    learning_rates = [0.001]#, 0.001, 0.01, 0.1, 0.0001]
    epochs = 10
    logs_path = ".tmp/orl/2"

    n_input = 10304
    n_hidden_1 = 256
    n_hidden_2 = 256
    n_classes = 20

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    dropout = 0.75  # Dropout, probability to keep units
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout, enableDropout=True):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 92, 112, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        if enableDropout: fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([23 * 28 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    for learning_rate in learning_rates:
        # Construct model
        logits = conv_net(X, weights, biases, keep_prob, enableDropout=True)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('cross_entropy_orl', loss_op)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            losses = []
            class_errors = []

            # perform training cycles
            for epoch in range(epochs):
                num_batches = int(240 / batch_size)
                epoch_accuracy = 0
                batch_loss = 0
                class_error = 0

                for i in range(num_batches):
                    x_train, y_train = trainX[i * batch_size:(i + 1) * batch_size], trainY[
                                                                                    i * batch_size:(i + 1) * batch_size]
                    # print x_train.shape, y_train.shape

                    # perform the operations we defined earlier on batch
                    _, loss, acc, summary = sess.run([train_op, loss_op, accuracy, merged], feed_dict={X: x_train, Y: y_train, keep_prob: dropout})
                    writer.add_summary(summary, epoch * num_batches + i)
                    epoch_accuracy += acc
                    class_error += 1.0 - acc

                    batch_loss += loss

                batch_loss = batch_loss/num_batches
                epoch_accuracy /= num_batches
                losses += [batch_loss]
                class_errors += [1.0-epoch_accuracy]

                print "Epoch: ", epoch, " Accuracy: ", epoch_accuracy, " Loss: ", batch_loss
                # print losses
            print learning_rate
            plt.plot(range(epochs), class_errors)
            plt.show()
            print "Test accuracy: ", accuracy.eval(feed_dict={X: testX, Y: testY})

            print "done"

            def visualise_conv_filter(weights, session):

                conv_filter = session.run(weights)  # get weights
                # sub plots grid layout
                sub_plots_grids = int(math.ceil(math.sqrt(conv_filter.shape[3])))

                # plot figure with sub-plots an grids.
                fig, axs = plt.subplots(sub_plots_grids, sub_plots_grids)

                for i, ax in enumerate(axs.flat):  # loop through all the filters
                    if i < conv_filter.shape[3]:  # check if a filter is valid
                        img = conv_filter[:, :, 0, i]  # format image
                        ax.imshow(img, vmin=np.min(conv_filter), vmax=np.max(conv_filter))  # plot image

                    # Remove marks from the axis
                    ax.set_yticks([])
                    ax.set_xticks([])

            visualise_conv_filter(weights["wc1"], sess)


if __name__ == '__main__':
    tf.app.run(main=main)
