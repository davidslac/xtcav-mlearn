from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tf_utils

########### support code - move to library? ###########
def inference(mod, x_input, hidden1_units, hidden2_units, activation_unit_name='tanh', weightsNorm = 'L2'):
    '''computes interence model for a 2 layer NN

    Args:
      mod:           model to update, adds attributes
      x_input:       N x 1 vector of input points.
      hidden1_units: makes a 1 x H1 set of weights for hidden layer, and a 
                             1 x H1 set of biases
      hidden2_units: makes a H1 x H2 set of weights for layer 2, and a 
                             1 X H2 set of biases
                     Then a final H2 x 1 set of weights for linear layer, and one bias
      activation_unit_name: activation function for hidden layer connections
      weightsNorm:   'L2' or 'L1' to form regularization penalty terms from weights

    Output:
       mod:   the model passed in with these additional fields:
    '''
    activation_unit = tf_utils.getActivationFunction(activation_unit_name)
    normPenalty = tf_utils.getNormPenalty(weightsNorm)

    mod.W_regterms = []

    #Hidden 1
    hiddenShape1 = [1, hidden1_units]
    mod.H1_W = tf.Variable(tf.truncated_normal(hiddenShape1,stddev=0.10), name='H1_W')
    mod.H1_B = tf.Variable(tf.zeros([hidden1_units]), name='H1_B')
    mod.H1_O = activation_unit(tf.nn.xw_plus_b(x_input, mod.H1_W, mod.H1_B))
    mod.W_regterms.append(normPenalty(mod.H1_W, name='H1_W_regterm'))

    #Hidden 2
    hiddenShape2 = [hidden1_units, hidden2_units]
    mod.H2_W = tf.Variable(tf.truncated_normal(hiddenShape2, stddev=0.1), name='H2_W')
    mod.H2_B = tf.Variable(tf.zeros([hidden2_units]), name='H2_B')
    mod.H2_O = activation_unit(tf.nn.xw_plus_b(mod.H1_O, mod.H2_W, mod.H2_B))
    mod.W_regterms.append(normPenalty(mod.H2_W, name='H2_W_regterm'))

    # Linear
    mod.L_W = tf.Variable(tf.truncated_normal([hidden2_units, 1], stddev=0.1), name='L_W')
    mod.L_B = tf.Variable(tf.zeros([1]), name='L_B')
    mod.nnetModel = tf.nn.xw_plus_b(mod.H2_O, mod.L_W, mod.L_B)

    # do you want to put the linear output layer into the regularization term?
    mod.W_regterms.append(normPenalty(mod.L_W, name='L_W_regtrem'))
        
    return mod


def loss(nnetModel, W_regterms, regFactors, labels):
    modelError = tf.reduce_mean(tf.square(tf.sub(nnetModel,labels)), name='modelError')
    # make scalar term
    regularizationTerm = tf.Variable(tf.zeros([], dtype=np.float32), name='regterm')
    if isinstance(regFactors, float) or isinstance(regFactors, int):
        regFactors= [regFactors for w in W_regterms]
    regFactors = np.array(regFactors).astype(np.float32)
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
   

