# Tensorflow Logistic Regression. Modified from the Tensorflow Logistic Regression
# Tutorial example code
import tensorflow as tf
import numpy as np

class Logistic_Regression:
    def __init__(self, input_dims = [784], output_dims = [10], lr_npy_path=None, trainable=True):
        if lr_npy_path is not None:
            try:
                self.data_dict = np.load(lr_npy_path, encoding='latin1').item()
                print("Loading Weights from File")
            except FileNotFoundError:
                print("Unable to load Weight File")
                self.data_dict = None
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.input_dims = input_dims
        self.output_dims = output_dims

        w_dims = list(input_dims.copy())
        w_dims.extend(output_dims)
        # Set model weights
        w_init = tf.zeros(w_dims)
        self.W = self.get_var(w_init, 'W', 0, 'weights')
        self.b = self.get_var(tf.zeros(output_dims), 'b', 0, 'biases')

    def build(self, x):
        self.W = tf.reshape(self.W,[self.input_dims[-1], -1])
        tmp = tf.matmul(x, self.W)
        tmp = tf.reshape(tmp, self.output_dims) 
        self.pred = tf.nn.softmax(tmp + self.b) # Softmax

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
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def cross_entropy_loss(self, y):
        #p = tf.reduce_sum(self.pred) #, reduction_indices = 1)
        return tf.reduce_mean(-tf.reduce_sum(tf.matmul(y,tf.log(self.pred)), reduction_indices=1))

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
    lr = Logistic_Regression(input_dims = [784], output_dims = [10,2], lr_npy_path = 'lr-save.npy')
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
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print ("Optimization Finished!")
        # Test model
        correct_prediction = tf.equal(tf.argmax(lr.pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        print ("Saving")
        lr.save_npy(sess)

if __name__ == '__main__':
    train_mnist()
