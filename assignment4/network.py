from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

# Setup verbosity
tf.logging.set_verbosity(tf.logging.INFO)

######### Macros and Hyper-parameters #########
learning_rate = 0.001
epochs = 100

batch_size = 24
num_input = 92 * 112
n_classes = 20  # Data has faces of 20 diff people
p_dropout = 0.04  # Dropout, probability to drop

# Nitty Gritties | format - (layer1_val, layer2_val)
strides = (2, 2)
filters = (32, 64)
kernels = (5, 5)
shape = (92, 112)
dense_units = 200


# Function to represent the model
def model(features, labels, mode):
    """
    Following tf boilerplate because tf is a fundamentalist fuck and I hate it but it works. (Sums up google, eh)

    :param features: lord knows
    :param labels: one-hot numpy arrays
    :param mode: scientists are still in the process of discovering
    :return: the necter of gods
    """

    """
        Input layer.

        OP Dim - [-1, 92, 112, 1]
    """
    input_vars = tf.reshape(features["x"], [-1, 92, 112, 1])  # Reshape X to 4-D tensor: [batch_size, width, height, channels]

    """
        Convolution layer 1.

            Connected with input_vars
        IP Dim - [-1, 92, 112, 1]
        OP Dim - [-1, 46, 56, 32]    (dim/pool_stride)
    """
    layer1_conv = tf.layers.conv2d(  # Convolution
        inputs=input_vars,
        filters=filters[0],
        kernel_size=[kernels[0], kernels[0]],
        padding="same",
        activation=tf.nn.relu
    )

    layer1_maxpool = tf.layers.max_pooling2d(  # Pooling after Convolution
        inputs=layer1_conv,
        pool_size=strides[0],
        strides=strides[0]
    )

    """
        Convolution layer 2.

            Connected with layer1_maxpool
        IP Dim - [-1, 46, 56, 1]
        OP Dim - [-1, 23, 28, 1]   (dim/pool_stride^2)
    """
    layer2_conv = tf.layers.conv2d(  # Convolution
        inputs=layer1_maxpool,
        filters=filters[1],
        kernel_size=[kernels[1], kernels[1]],
        padding="same",
        activation=tf.nn.relu
    )

    layer2_maxpool = tf.layers.max_pooling2d(  # Pooling after Convolution
        inputs=layer2_conv,
        pool_size=strides[1],
        strides=strides[1]
    )

    """
        Flatten

            Take this maxpool o/p, flatten it
        IP Dim - [-1, 23, 28, 64]
        OP Dim - [-1, 23* 28* 64]
    """
    flattened_layer = tf.reshape(layer2_maxpool, [-1, 23*28*64])

    """
        MLP Time

            Dense; Dropout; Dense
        IP Dim - [ -1, 23* 28* 64]
        OP Dim - [ -1, 23* 28* 64]
    """
    dropout = tf.layers.dropout(inputs=flattened_layer, rate=p_dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense = tf.layers.dense(inputs=dropout, units=dense_units, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=dense, units=n_classes)

    # One hot the labels
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)

    # Calculate Loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output)

    # More boilerplate code IDK what to do with
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=output, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(output, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        # "accuracy": tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))}
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(useless_boilerplate_unknown_param):

    # Pull the data from the disk
    data = np.load('data/ORL_faces.npz')
    X_train = data['trainX'].astype("float32")
    Y_train = data['trainY'].astype("int32")
    X_test = data['testX'].astype("float32")
    Y_test = data['testY'].astype("int32")

    # # One hot encode the y labels
    # one_hot_matrix = np.eye(n_classes)
    # Y_train = one_hot_matrix[Y_train].astype("int32")
    # Y_test = one_hot_matrix[Y_test].astype("int32")

    # Create estimator, plug the model fn there
    estimator = tf.estimator.Estimator(
        model_fn=model,
        model_dir=".tmp/mnist"
    )

    # Logging boilerplate
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    estimator.train(
        input_fn=train_input_fn,
        steps=epochs,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=Y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":

    tf.app.run()