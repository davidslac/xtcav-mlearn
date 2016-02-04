from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tf_utils
import function_fit as ffit
import tensorflowH5 as tfh5
from collections import namedtuple
import h5py

class Bag(object):
    pass

FLAGS = Bag()
FLAGS.learning_rate =  0.01 # "initial learning rate"
FLAGS.max_steps =  500 # 'Numer of steps to run trainer'
FLAGS.num_eval_updates =  50 # "Number of evaluations to do during training"
FLAGS.eval_steps =  FLAGS.max_steps//FLAGS.num_eval_updates # 'steps between evaluations'
FLAGS.hidden1 =  100 # 'Number of units in hidden layer 1.'
FLAGS.hidden2 =  5 # 'Number of units in hidden layer 2.'
FLAGS.actfn =  'relu' # 'activation function - one of tanh, sigmoid, softplus relu relu6
FLAGS.regNorm =  'L2' # 'regularization norm function'
FLAGS.poly_roots =  7 # 'Number of polynomial roots.'
FLAGS.reg = 1.0
FLAGS.reg_hidden1 =  FLAGS.reg # "regularization term for hidden layer 1"
FLAGS.reg_hidden2 =  FLAGS.reg # "regularization term for hidden layer 2"
FLAGS.reg_output =  FLAGS.reg # "regularization term for linear output"
FLAGS.train_dir =  'train_dir' # "directory to write training data"
FLAGS.summaryfile = 'train_dir/run_summary.h5'
FLAGS.numberCores = 4

def main():
    # get training/test data
    polyData = tf_utils.getPolyData(FLAGS.poly_roots)
    mod = Bag()
                            
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        mainAfterWithGraph(polyData, mod)

def mainAfterWithGraph(polyData, mod):
    # Generate placeholders for the input and labels
    placeholder = namedtuple('PlaceHolder', 'x_input y_output')
    placeholder.x_input = tf.placeholder(tf.float32, shape=(None, 1))
    placeholder.y_output = tf.placeholder(tf.float32, shape=(None, 1))

    mod.placeholder = placeholder

    # Build a Graph that computes predictions from the inference model.
    mod = ffit.inference(mod, mod.placeholder.x_input, FLAGS.hidden1, FLAGS.hidden2, 
                   FLAGS.actfn, FLAGS.regNorm)

    # Add to the Graph the Ops for loss calculation.
    mod.meanSqError, mod.loss = ffit.loss(mod.nnetModel, mod.weightsPenaltyTerms,
                                          [FLAGS.reg_hidden1, FLAGS.reg_hidden2, FLAGS.reg_output],
                                          mod.placeholder.y_output)

    # Add to the Graph the Ops that calculate and apply gradients.
    mod.train_op = ffit.training(mod.loss, FLAGS.learning_rate)
    
    # Build the summary operation based on the TF collection of Summaries.
#    mod.train_summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
#    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()

    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
 #   summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
 #                                           graph_def=sess.graph_def)

    h5out = tfh5.TensorFlowH5Writer(FLAGS.summaryfile)

    train_feed_dict = {mod.placeholder.x_input: polyData.x_train,
                       mod.placeholder.y_output:polyData.y_train}

    test_feed_dict = {mod.placeholder.x_input: polyData.x_test,
                      mod.placeholder.y_output:polyData.y_test}

    all_feed_dict = {mod.placeholder.x_input: polyData.x_all,
                     mod.placeholder.y_output:polyData.y_all}

    # And then after everything is built, start the training loop.
#    saver.restore(sess, FLAGS.train_dir)

    for step in xrange(FLAGS.max_steps):
        trainStep(step, mod, sess, h5out, train_feed_dict, test_feed_dict)


def trainStep(step, mod, sess, h5out, train_feed_dict, test_feed_dict):
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, trainLoss = sess.run([mod.train_op, mod.loss],
                             feed_dict=train_feed_dict)

    # Write the summaries and print an overview fairly often.
    if step % FLAGS.eval_steps == 0:
        trainAvgErr = sess.run(mod.meanSqError, feed_dict=train_feed_dict)
        testAvgErr = sess.run(mod.meanSqError, feed_dict=test_feed_dict)
        h5out.addNumpyArrays(step, [trainAvgErr, testAvgErr, trainLoss],
                             ['trainAvgErr', 'testAvgErr', 'trainLoss'])
        h5out.addTensors(sess, step, [mod.hidden1_weights, 
                                      mod.hidden1_biases, 
                                      mod.hidden2_weights,
                                      mod.hidden2_biases,
                                      mod.linear_weights, 
                                      mod.linear_bias],
                                     ['hidden1_weights', 
                                      'hidden1_biases', 
                                      'hidden2_weights',
                                      'hidden2_biases',
                                      'linear_weights', 
                                      'linear_bias'])
        # Print status to stdout.
        print('Step %d: loss = %.2f trainErr=%.2f testErr=%.2f' % (step, trainLoss, trainAvgErr, testAvgErr))
        # Update the events file.
#        summary_str = sess.run(summary_op, feed_dict=feed_dict)
#        summary_writer.add_summary(summary_str, step)

 
def plotSummaryFile(fname):
    import matplotlib.pyplot as plt
    h5plt = tfh5.TensorFlowH5Plotter(fname, plt)
    h5plt.scalarSummary(['trainAvgErr', 'testAvgErr', 'trainLoss'])
    plt.figure()
    h5plt.singleInputWeightBiases('hidden1_weights', 'hidden1_biases')
    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
    plotSummaryFile(FLAGS.summaryfile)
