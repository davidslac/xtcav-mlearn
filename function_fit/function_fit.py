from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tf_utils

########### support code - move to library? ###########
def inference(x_input, hidden1_units, hidden2_units, activation_unit_name='tanh', weightsNorm = 'L2'):
    '''computes interence model for a 2 layer NN

    Args:
      x_input:       N x 1 vector of input points.
      hidden1_units: makes a 1 x H1 set of weights for hidden layer, and a 
                             1 x H1 set of biases
      hidden2_units: makes a H1 x H2 set of weights for layer 2, and a 
                             1 X H2 set of biases
                     Then a final H2 x 1 set of weights for linear layer, and one bias
      activation_unit_name: activation function for hidden layer connections
      weightsNorm:   'L2' or 'L1' to form regularization penalty terms from weights

    Output:
       nnetModel  - the nnetModel that computes inference from x_input
       weightsPenaltyTerms - a list of three weight penalty terms for regularization,
                             hidden1, hidden2, then the linear layer
    '''
    activation_unit = tf_utils.getActivationFunction(activation_unit_name)
    normPenalty = tf_utils.getNormPenalty(weightsNorm)

    weightsPenaltyTerms = []

    #Hidden 1
    with tf.name_scope('hidden1'):
        hiddenShape = [1, hidden1_units]
        weights = tf.Variable(
            tf.truncated_normal(hiddenShape,
                                stddev=0.10),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                            name='biases')
        hidden1 = activation_unit(tf.nn.xw_plus_b(x_input, weights,biases))
        weightsPenaltyTerms.append(normPenalty(weights, name='l2-weights'))

    #Hidden 2
    with tf.name_scope('hidden2'):
        hiddenShape = [hidden1_units, hidden2_units]
        weights = tf.Variable(
            tf.truncated_normal(hiddenShape,
                                stddev=0.1),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = activation_unit(tf.nn.xw_plus_b(hidden1, weights, biases))
        weightsPenaltyTerms.append(normPenalty(weights, name='l2-weights'))

    # Linear
    with tf.name_scope('linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, 1],
                                stddev=0.1),
            name='weights')
        biases = tf.Variable(tf.zeros([1]), name='biases')
        nnetModel = tf.nn.xw_plus_b(hidden2, weights, biases)
        # do you want to put the linear output layer into the regularization term?
        weightsPenaltyTerms.append(normPenalty(weights, name='l2-weights'))

    return nnetModel, weightsPenaltyTerms



def loss(nnetModel, weightPenTermList, regHyperParamList, labels):
    modelError = tf.reduce_mean(tf.pow(tf.sub(nnetModel,labels),2), name='model-error')
    # make scalar term
    regularizationTerm = tf.Variable(tf.zeros([], dtype=np.float32), name='regterm')
    if isinstance(regHyperParamList, float) or isinstance(regHyperParamList, int):
        regHyperParamList = [regHyperParamList for w in weightPenTermList]
    regHyperParamList = np.array(regHyperParamList).astype(np.float32)
    for wT, hyperParam in zip(weightPenTermList, regHyperParamList):
        regularizationTerm += tf.mul(hyperParam, wT)
    modelLoss = tf.add(modelError, regularizationTerm, name='loss')
    return modelError, modelLoss


def training(loss, learning_rate):
    # http://stackoverflow.com/questions/35119109/tensorflow-scalar-summary-tags-name-exception
    # this has created problems, for scalar_summary, the shape of the name has to match the
    # shape of the item that we are viewing. If loss involves a vector term, i.e
    # regTerm = Variable([0]) ...  # this defines a vector of shape 1.
    # all the other scalar terms get broadcast to this vector of shape 1,
    # in which case the one would have to match it with 
    # tf.scalar_summary([loss.op.name], loss)

    # I think what we really want is a scalar, so be more careful with defining regTerm
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
   
def evaluation(sess, nnetModel, x_input, polyData):
    """Evaluate the model on the function data
    Args:
      nnetModel: tensor that takes x_input, produces y_output
      functionData: has x_test, y_test, train and all vectors
    Returns:
      meanSqRtErrorTrain  scalar float32 tensor 
      meanSqRtErrorTest   scalar float32
      y_all          vector, evaluate on all
    """
    def getSqRtError(nnetModel, x, y):
        mod_minus_y = tf.sub(nnetModel, y)
        squareError = tf.pow(mod_minus_y, 2)
        meanErr = tf.reduce_mean(squareError)
        return sess.run(tf.pow(meanErr,0.5), feed_dict={x_input:x})

    x_test = tf_utils.vec2columnMat(polyData.x_test)
    y_test = tf_utils.vec2columnMat(polyData.y_test)
    x_train = tf_utils.vec2columnMat(polyData.x_train)
    y_train = tf_utils.vec2columnMat(polyData.y_train)
    x_all =  tf_utils.vec2columnMat(polyData.x_all)

    trainErr = getSqRtError(nnetModel, x_train, y_train)
    testErr = getSqRtError(nnetModel, x_test, y_test)
    y_all = sess.run(nnetModel, feed_dict={x_input:x_all})
    return trainErr, testErr, y_all

