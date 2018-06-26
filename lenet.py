"""
LeNet (Yann LeCunn Developed 1st breed of Neural Network)
"""

import tensorflow as tf
import numpy as np
import cv2
import os


def LeNet(x, keep_prob=1.0):
    mu = 0
    sigma = 0.1

    print("Single scale LeNet", " with dropout" if keep_prob != 1.0 else "")

    with tf.variable_scope('P0'):
        p0 = tf.image.convert_image_dtype(x, tf.float32)
        p0 = tf.divide(tf.subtract(p0, 128.0), 128.0)
        print("P0: Input %s Output %s" % (x.get_shape(), p0.get_shape()))

    _, w, h, c = p0.get_shape().as_list()

    # C1: Input 32x32xc, Output: 28x28x6
    with tf.variable_scope('C1'):
        weight1 = tf.Variable(tf.truncated_normal(shape=(5, 5, c, 6), mean=mu, stddev=sigma))
        bias1 = tf.Variable(tf.zeros(shape=(6)))
        conv1 = tf.nn.conv2d(p0, weight1, strides=(1, 1, 1, 1), padding='VALID')
        conv1 = tf.add(conv1, bias1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
        print("C1: Input %s Output %s" % (x.get_shape(), conv1.get_shape()))

    # S2: Input 28x28x6, Output: 14x14x6
    with tf.variable_scope('S2'):
        pool1 = tf.nn.max_pool(conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        print("S2: Input %s Output %s" % (conv1.get_shape(), pool1.get_shape()))

    # C3: Input 14x14x6, Output: 10x10x16
    with tf.variable_scope('C3'):
        weight2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        bias2 = tf.Variable(tf.zeros(shape=(16)))
        conv2 = tf.nn.conv2d(pool1, weight2, strides=(1, 1, 1, 1), padding='VALID')
        conv2 = tf.add(conv2, bias2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)
        print("C3: Input %s Output %s" % (pool1.get_shape(), conv2.get_shape()))

    # S4: Input 10x10x16, Output 5x5x16
    with tf.variable_scope('S4'):
        pool2 = tf.nn.max_pool(conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        print("S4: Input %s Output %s" % (conv2.get_shape(), pool2.get_shape()))

    # C5: Input 5x5x16, Output 1x120
    with tf.variable_scope('C5'):
        fc1 = tf.contrib.layers.flatten(pool2)
        weight3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        bias3 = tf.Variable(tf.zeros(shape=(120)))
        fc1 = tf.matmul(fc1, weight3)
        fc1 = tf.add(fc1, bias3)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
        print("C5: Input %s Output %s" % (pool2.get_shape(), fc1.get_shape()))

    # F6: Input 1x120, Output 1x84
    with tf.variable_scope('F6'):
        weights4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        bias4 = tf.Variable(tf.zeros(shape=(84)))
        fc2 = tf.matmul(fc1, weights4)
        fc2 = tf.add(fc2, bias4)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
        print("F6: Input %s Output %s" % (fc1.get_shape(), fc2.get_shape()))

    # F7: Input 1x84, Output 1x43
    with tf.variable_scope('F7'):
        weight5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
        bias5 = tf.Variable(tf.zeros(shape=(43)))
        logits = tf.matmul(fc2, weight5)
        logits = tf.add(logits, bias5)
        print("F7: Input %s Output %s" % (fc2.get_shape(), logits.get_shape()))

    return logits


def train(X_train, y_train, learning_rate=0.001, epochs=10, batch_size=128, save_graph="lenet.pb"):
    # Extract input information
    height, width, channels = X_train[0].shape
    nclasses = len(np.unique(y_train))

    # Graph Nodes
    features = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='features')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    keep_prob = tf.placeholder(tf.float32)

    logits = LeNet(x, keep_prob)

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=832289)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss_op = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss_op, var_list=[weight8, bias8])

    preds = tf.argmax(logits, 1)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    model_dir = os.path.join(os.path.curdir, os.path.dirname(save_graph))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        total_loss = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy, loss = sess.run([accuracy_op, loss_op], feed_dict={features: batch_x, labels: batch_y})
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))
        return total_accuracy / num_examples, total_loss / num_examples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training ....")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                sess.run(training_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

            training_accuracy, training_loss = evaluate(X_train, y_train)
            validation_accuracy, validation_loss = evaluate(X_valid, y_valid)

            print("EPOCH {} ...".format(i+1))
            print("Training Accuracy = {:.3f} Loss = {:.3f}".format(training_accuracy, training_loss))
            print("Validation Accuracy = {:.3f} Loss = {:.3f}".format(validation_accuracy, validation_loss))
            print()

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["F7/Add"])

        with tf.gfile.GFile(os.path.join(model_dir, save_graph), "wb") as f:
            f.write(output_graph_def.SerializeToString())

def inference(fname, model, labels):
    """
    Args
        fname (str) - image filepath (*.jpg)
        model (str) - model filepath (*.pb)
        labels(str) - labels filepath (*.txt)

    Return
        predicted_label (str) - prediction result
    """
    import cv2
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    image = cv2.imread(fname)
    assert image is not None, "Failed to open [%s]" % (fname)

    input_tensor = graph.get_tensor_by_name("import/features:0")
    output_tensor = graph.get_tensor_by_name("import/F7/Add:0")

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(image, 0)})
    results = np.squeeze(predictions)

    top_k = results.argsort()[-5:][::-1]
    class_names = [l.strip() for l in tf.gfile.GFile(labels).readlines()]
    for i in top_k:
        print(class_names[i], results[i])

    return results[top_k[0]]


if __name__ == "__main__":

    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_p", default="train.p", help="path to train.p")
    parser.add_argument("--test_p", default="test.p", help="path to test.p")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    parser.add_argument("--epochs", type=int, default=2, help="number of training iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--train", action="store_true", help="train/retrain model")
    parser.add_argument("--save_graph", default="lenet.pb", help="saves graph as *.pb while training or loads *.pb for inference")
    parser.add_argument("--labels", default="labels.txt", help="labels file")
    parser.add_argument("--infer", default=None, help="perform inference for file")
    args = parser.parse_args()

    print(args)

    trainset = pickle.load(open(args.train_p, "rb"))
    testset = pickle.load(open(args.test_p, "rb"))

    X_train, y_train = trainset['features'], trainset['labels']
    X_test, y_test = testset['features'], testset['labels']

    if args.train:
        train(X_train, y_train, net_weights=args.net_weights, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate, save_graph=args.save_graph)

    if args.infer:
        inference(args.infer, args.save_graph, args.labels)
