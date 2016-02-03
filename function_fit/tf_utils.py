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
    def l2pen(weights, name):
        return tf.mul(1.0/2.0, tf.reduce_sum(weights), name=name)

    def l1pen(weights, name):
        return tf.reduce_sum(tf.abs(weights), name=name)
        
    normName2foo = {'L1':l1pen, 'L2':l2pen}

    assert normName in normName2foo, \
        "not a known norm name for penealty term, use one of %s" % normName2foo.keys()
    
    return normName2foo[normName]

    
def getPolyData(poly_degree=7, numPoints = 300, numTrain = None):
    assert poly_degree > 0, "must have at least 1 root"
    if numTrain is None:
        numTrain = max(1,int(.1 * numPoints))
    assert 3*numTrain < numPoints, "numTrain must be < 3*numPoints to produce test/train/cv sets"
    # make a polynomial with the given number of roots, but no root at 0
    poly_roots = list(np.arange(poly_degree + 1) - int(poly_degree//2))
    poly_roots.remove(0)
    poly_roots = np.array(poly_roots)
    poly_coeffs = polynomial.polyfromroots(poly_roots)

    polyData = namedtuple("PolyData", "x_all y_all x_train y_train x_test y_test x_cv y_cv")

    polyData.x_all = vec2columnMat(np.linspace(start = poly_roots[0],
                                               stop =  poly_roots[-1],
                                               num = 300))
    polyData.y_all = polynomial.polyval(polyData.x_all[:], poly_coeffs)
    inds = range(len(polyData.x_all))
    random.shuffle(inds)
    polyData.x_train = vec2columnMat(polyData.x_all[:][inds[0:numTrain]])
    polyData.y_train = vec2columnMat(polyData.y_all[:][inds[0:numTrain]])
    polyData.x_test = vec2columnMat(polyData.x_all[:][inds[numTrain:(2*numTrain)]])
    polyData.y_test = vec2columnMat(polyData.y_all[:][inds[numTrain:(2*numTrain)]])
    polyData.x_cv = vec2columnMat(polyData.x_all[:][inds[(2*numTrain):(3*numTrain)]])
    polyData.y_cv = vec2columnMat(polyData.y_all[:][inds[(2*numTrain):(3*numTrain)]])
    return polyData

def plotPolyData(polyData, plt, figH=None):
    if figH is None:
        plt.figure()
    else:
        plt.figure(figH)

    plt.plot(polyData.x_all, polyData.y_all, label='all')
    plt.plot(polyData.x_train, polyData.y_train, '*', label='train')
    plt.plot(polyData.x_test, polyData.y_test, 'o', label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    polyData = getPolyData(5)
    import matplotlib.pyplot as plt
    plotPolyData(polyData, plt)
