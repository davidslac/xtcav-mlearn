from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.polynomial.polynomial as polynomial
import tensorflow as tf
import random
from collections import namedtuple

def vec2columnMat(vec):
    n = len(vec)
    return np.resize(vec,(n,1))

def getActivationFunction(name):
    activation_name2function = {'sigmoid': tf.nn.sigmoid,
                                'tanh': tf.nn.tanh,
                                'softplus': tf.nn.softplus,
                                'relu':tf.nn.relu,
                                'relu6':tf.nn.relu6}

    assert name in activation_name2function, \
        "activation unit name must be one of %s" % activation_name2function.keys()

    return activation_name2function[name]


def getNormPenalty(normName):
    def l2pen(tensor, name):
        return tf.mul(0.5, tf.reduce_sum(tf.square(tensor)), name=name)

    def l1pen(tensor, name):
        return tf.reduce_sum(tf.abs(tensor), name=name)
        
    normName2foo = {'L1':l1pen, 'L2':l2pen}

    assert normName in normName2foo, \
        "not a known norm name for penealty term, use one of %s" % normName2foo.keys()
    
    return normName2foo[normName]

def getFlattendGrad(sess, grads_and_vars, feed_dict):
    grads = sess.run([tv[0] for tv in grads_and_vars], feed_dict=feed_dict)
    grads = [grad.flatten() for grad in grads]
    return np.concatenate(grads)    

class FunctionData(object):
    def __init__(self, poly_degree=7, numPoints = 300, numTrain = None, h5=None):
        if h5 is not None:
            self.h5read(h5)
            return
        assert poly_degree > 0, "must have at least 1 root"
        if numTrain is None:
            numTrain = max(1,int(.1 * numPoints))
        assert 3*numTrain < numPoints, "numTrain must be < 3*numPoints to produce test/train/cv sets"
        # make a polynomial with the given number of roots, but no root at 0
        poly_roots = list(np.arange(poly_degree + 1) - int(poly_degree//2))
        poly_roots.remove(0)
        poly_roots = np.array(poly_roots)
        poly_coeffs = polynomial.polyfromroots(poly_roots)

        self.x_all = vec2columnMat(np.linspace(start = poly_roots[0],
                                                   stop =  poly_roots[-1],
                                                   num = 300))
        self.y_all = polynomial.polyval(self.x_all[:], poly_coeffs)
        inds = range(len(self.x_all))
        random.shuffle(inds)
        self.x_train = vec2columnMat(self.x_all[:][inds[0:numTrain]])
        self.y_train = vec2columnMat(self.y_all[:][inds[0:numTrain]])
        self.x_test = vec2columnMat(self.x_all[:][inds[numTrain:(2*numTrain)]])
        self.y_test = vec2columnMat(self.y_all[:][inds[numTrain:(2*numTrain)]])
        self.x_cv = vec2columnMat(self.x_all[:][inds[(2*numTrain):(3*numTrain)]])
        self.y_cv = vec2columnMat(self.y_all[:][inds[(2*numTrain):(3*numTrain)]])

    def h5write(self, h5, groupName='FunctionData'):
        gr = h5.create_group(groupName)
        gr['x_all'] = self.x_all
        gr['y_all'] = self.y_all
        gr['x_train'] = self.x_train
        gr['y_train'] = self.y_train
        gr['x_test'] = self.x_test
        gr['y_test'] = self.y_test
        gr['x_cv'] = self.x_cv
        gr['y_cv'] = self.y_cv

    def h5read(self, h5, groupName='FunctionData'):
        gr = h5[groupName]
        self.x_all = gr['x_all'] 
        self.y_all = gr['y_all'] 
        self.x_train = gr['x_train'] 
        self.y_train = gr['y_train'] 
        self.x_test = gr['x_test'] 
        self.y_test = gr['y_test'] 
        self.x_cv = gr['x_cv'] 
        self.y_cv = gr['y_cv'] 
        
    def plot(self, plt):
        plt.plot(self.x_all, self.y_all, label='all')
        plt.plot(self.x_train, self.y_train, '*', label='train')
        plt.plot(self.x_test, self.y_test, 'o', label='test')
        plt.legend()

if __name__ == '__main__':
    functionData = getFunctionData(5)
    import matplotlib.pyplot as plt
    plotFunctionData(functionData, plt)
