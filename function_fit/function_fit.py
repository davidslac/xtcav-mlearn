from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tf_utils

########### support code - move to library? ###########
def inference(mod, x_input, layerNodes, activation_unit_name='tanh', weightsNorm = 'L2', stddev=0.1):
    '''computes interence model for a N layer NN that fits function, a 1D -> 1D function

    Args:
      mod:           model to update, adds attributes
      x_input:       N x 1 vector of input points.
      layerNodes:    list of number of hidden units.
      activation_unit_name: activation function for hidden layer connections
      weightsNorm:   'L2' or 'L1' to form regularization penalty terms from weights

    Output:
       mod:   the model passed in with these additional fields:
    '''
    activation_unit = tf_utils.getActivationFunction(activation_unit_name)
    normPenalty = tf_utils.getNormPenalty(weightsNorm)

    mod.W_regterms = []
    previousHiddenUnits = 1
    prevUnits = x_input

    # note: this loop won't execute for a linear model:
    for idx,hiddenUnits in enumerate(layerNodes):
        layerNumber = idx+1
        hiddenShape = [previousHiddenUnits, hiddenUnits]
        nameW = 'H%02d_W' % layerNumber
        nameB = 'H%02d_B' % layerNumber
        nameU = 'H%02d_U' % layerNumber
        nameReg = 'H%02d_W_regterm' % layerNumber
        Weights = tf.Variable(tf.truncated_normal(hiddenShape,stddev=stddev), name=nameW)
        regW = normPenalty(Weights, name=nameReg)
        Bias = tf.Variable(tf.zeros([hiddenUnits]), name=nameB)
        Units = activation_unit(tf.nn.xw_plus_b(prevUnits, Weights, Bias))
        setattr(mod,nameW,Weights)
        setattr(mod,nameB,Bias)
        setattr(mod,nameU,Units)
        mod.W_regterms.append(regW)
        previousHiddenUnits = hiddenUnits
        prevUnits = Units

    # Linear
    mod.L_W = tf.Variable(tf.truncated_normal([previousHiddenUnits, 1], stddev=stddev), name='L_W')
    mod.L_B = tf.Variable(tf.zeros([1]), name='L_B')
    mod.nnetModel = tf.nn.xw_plus_b(prevUnits, mod.L_W, mod.L_B)

    return mod


def loss(nnetModel, W_regterms, regFactors, labels):
    modelError = tf.reduce_mean(tf.square(tf.sub(nnetModel,labels)), name='modelError')
    # make scalar term
    regularizationTerm = tf.Variable(tf.zeros([], dtype=np.float32), name='regterm')
    regFactors = np.array(regFactors).astype(np.float32)
    assert len(regFactors) >= len(W_regterms), "to few regFactors for %d regterms" % len(W_regterms)
    for wT, regFact in zip(W_regterms, regFactors):
        regularizationTerm += tf.mul(regFact, wT)
    modelLoss = tf.add(modelError, regularizationTerm, name='loss')
    return modelError, modelLoss


def training(mod, loss, learning_rate):
    # http://stackoverflow.com/questions/35119109/tensorflow-scalar-summary-tags-name-exception
    # this has created problems, for scalar_summary, the shape of the name has to match the
    # shape of the item that we are viewing. If loss involves a vector term, i.e
    # regTerm = Variable([0]) ...  # this defines a vector of shape 1.
    # all the other scalar terms get broadcast to this vector of shape 1,
    # in which case the one would have to match it with 
    # tf.scalar_summary([loss.op.name], loss)

    # I think what we really want is a scalar, so be more careful with defining regTerm
#    tf.scalar_summary(loss.op.name, loss)

    # Create the gradient descent optimizer with the given learning rate.
    mod.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step.
    mod.global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
#    mod.train_op = mod.optimizer.minimize(loss, global_step=mod.global_step)

    # break out the minimization so that we can see 
    # https://www.tensorflow.org/versions/master/api_docs/python/train.html#Optimizer.minimize
    mod.grads_and_vars = mod.optimizer.compute_gradients(loss)
    mod.train_op = mod.optimizer.apply_gradients(mod.grads_and_vars)
    return mod
   

