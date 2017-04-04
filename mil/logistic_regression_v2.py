# Tensorflow Logistic Regression. Modified from the Tensorflow Logistic Regression
# Tutorial example code
#
#  NOTE: CONVERTED FOR PYTHON 2
import tensorflow as tf
import numpy as np

INSTANCE_DIM = 512

class Logistic_Regression:
    def __init__(self, input_dims = [INSTANCE_DIM], output_dim = 1, lr_npy_path=None, trainable=True):
        if lr_npy_path is not None:
            try:
                self.data_dict = np.load(lr_npy_path, encoding='latin1').item()
                print("Loading Weights from File")
            except:
                print("Unable to load Weight File")
                self.data_dict = None
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

        w_dims = list(np.array(input_dims, copy=True))
        w_dims.append(output_dim)
        # Set model weights
        w_init = tf.zeros(w_dims)
        self.W = self.get_var(w_init, 'W', 0, 'weights')
        self.b = self.get_var(tf.zeros(output_dim), 'b', 0, 'biases')

    def build(self, x):
        assert x.shape == (2940, 512)
        self.r = tf.nn.sigmoid(tf.matmul(x, self.W) + self.b) # Softmax
        self.pred = tf.reduce_max(self.r)

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./lr-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print ("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def cross_entropy_loss(self, y):
        return -tf.log((self.pred*y) + (1 - self.pred) * (1-y))


from kaggle_data import *

LABELS = get_labels_by_name()

def get_flattened_vgg_features():
    for feats4D, lung_id in get_training_vgg_features():
        feats2D = feats4D
        yield feats4D.reshape((-1, INSTANCE_DIM)), lung_id


def train_mil():
    learning_rate = 0.01
    training_epochs = 5
    batch_size = 1
    display_step = 1

    x = tf.placeholder(tf.float32, [2940, INSTANCE_DIM]) 
    y = tf.placeholder(tf.float32, [1]) 

    # Construct model
    lr = Logistic_Regression(input_dims = [INSTANCE_DIM], output_dim = 10, lr_npy_path = 'lr-save.npy')
    lr.build(x)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Minimize error using cross entropy
        cost = lr.cross_entropy_loss(y)
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            num_lungs = 0
            # Loop over all lungs
            for lung, l_id in get_flattened_vgg_features():
                batch_xs, batch_ys =lung, np.array([LABELS[l_id]])
                feed_dict = {x : batch_xs, y: batch_ys}
                # Fit training using batch data
                pred = sess.run(lr.pred, feed_dict=feed_dict)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c
                num_lungs += 1
            avg_cost /= num_lungs
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{}".format(avg_cost))

        print "Optimization Finished!"
        # Test model
        correct_prediction = tf.equal(tf.argmax(lr.pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print "Saving"
        #lr.save_npy(sess)


def train_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    learning_rate = 0.01
    training_epochs = 5
    batch_size = 100
    display_step = 1

    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) 

    # Construct model
    lr = Logistic_Regression(input_dims = [784], output_dim = 10, lr_npy_path = 'lr-save.npy')
    lr.build(x)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Minimize error using cross entropy
        cost = lr.cross_entropy_loss(y)
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                print batch_xs.shape
                print batch_ys.shape
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print "Optimization Finished!"
        # Test model
        correct_prediction = tf.equal(tf.argmax(lr.pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print "Saving"
        #lr.save_npy(sess)

if __name__ == '__main__':
    #train_mnist()
    train_mil()
