
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import numpy as np
import numpy.polynomial.polynomial as polynomial
import tensorflow as tf
random.seed(234921)


# make a polynomial with the given number of roots, but no root at 0
poly_degree = 7
poly_roots = list(np.arange(poly_degree + 1) - int(poly_degree/2))
poly_roots.remove(0)
poly_roots = np.array(poly_roots)
poly_coeffs = polynomial.polyfromroots(poly_roots)

deltax = .1
x_all = np.arange(poly_roots[0]-.2, poly_roots[-1]+.2, deltax)
y_all = polynomial.polyval(x_all, poly_coeffs)

number_points_to_train = 20
number_points_to_test = 20
inds = range(len(x_all))
random.shuffle(inds)
x_train = x_all[inds[0:number_points_to_train]]
y_train = y_all[inds[0:number_points_to_train]]
x_test = x_all[inds[number_points_to_train:(number_points_to_train + number_points_to_test)]]
y_test = y_all[inds[number_points_to_train:(number_points_to_train + number_points_to_test)]]


# Nueral network

nnModel = namedtuple("TensorFlowNNModel", "X Y W_h B_h W_o B_o model avgError sumWeights regularizationTerm loss optimizer train init sess")
number_nuerons_hidden_layer = 500
stddev = 0.1
nnModel.X = tf.placeholder("float32", [None,1], name="x-input")
nnModel.Y = tf.placeholder("float32", [None,1], name="y-target")
nnModel.W_h = tf.Variable(tf.random_normal((1, number_nuerons_hidden_layer), stddev=stddev), name="weights-hidden-layer")
nnModel.B_h = tf.Variable(tf.random_normal((1, number_nuerons_hidden_layer), stddev=stddev), name="biases-hidden-layer")
nnModel.W_o = tf.Variable(tf.random_normal((number_nuerons_hidden_layer,1), stddev=stddev), name="weights-output-layer")
nnModel.B_o = tf.Variable(tf.random_normal((1,1), stddev=stddev), name="bias-ouput-layer")

nnModel.regularization = 0.1

# build model
X_times_W_h = tf.matmul(nnModel.X, nnModel.W_h)
add_bias_h = tf.add(X_times_W_h, nnModel.B_h)
hidden_layer_res = tf.nn.sigmoid(add_bias_h)
hidden_times_W_o = tf.matmul(hidden_layer_res, nnModel.W_o)
nnModel.model = tf.add(hidden_times_W_o, nnModel.B_o)

nnModel.avgError = tf.reduce_mean(tf.square(nnModel.model - nnModel.Y), name='avg-error')

nnModel.sumWeights = tf.reduce_sum(tf.mul(nnModel.W_h, nnModel.W_h)) + \
                     tf.reduce_sum(tf.mul(nnModel.W_o, nnModel.W_o), name='sum-activations')
nnModel.regularizationTerm = tf.mul(nnModel.regularization, nnModel.sumWeights)
                                  
nnModel.loss =  tf.add(nnModel.avgError, nnModel.regularizationTerm, name="loss")
nnModel.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
nnModel.train = nnModel.optimizer.minimize(nnModel.loss)

nnModel.init = tf.initialize_all_variables()
nnModel.sess = tf.Session()
nnModel.sess.run(nnModel.init)

nn_X_train = np.resize(x_train, (len(x_train),1))
nn_Y_train = np.resize(y_train, (len(y_train),1))

nn_X_test = np.resize(x_test, (len(x_test),1))
nn_Y_test = np.resize(y_test, (len(y_test),1))

for step in xrange(1001):
    nnModel.sess.run(nnModel.train, feed_dict={nnModel.X:nn_X_train, nnModel.Y:nn_Y_train})
    if step % 100 == 0:
        curLoss, curAvgError= nnModel.sess.run([nnModel.loss, nnModel.avgError], \
                                               feed_dict={nnModel.X:nn_X_train, nnModel.Y:nn_Y_train})
        curAvgTestError, curSumWeights= nnModel.sess.run([nnModel.avgError, nnModel.sumWeights], \
                                                         feed_dict={nnModel.X:nn_X_test, nnModel.Y:nn_Y_test})

        print "step=%d loss=%8.2f avgError=%8.2f avgTestError=%8.2f sumWeights=%12.1f" % \
            (step, curLoss, curAvgError, curAvgTestError, curSumWeights)

nnModel.sess.close()
del nnModel


