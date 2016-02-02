from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tf_utils
import function_fit as ffit

from collections import namedtuple
#import h5py

FLAGS = namedtuple("Params", 'learning_rate max_steps')
FLAGS.learning_rate =  0.01 # "initial learning rate"
FLAGS.max_steps =  500 # 'Numer of steps to run trainer'
FLAGS.num_eval_updates =  5 # "Number of evaluations to do during training"
FLAGS.eval_steps =  FLAGS.max_steps//FLAGS.num_eval_updates # 'steps between evaluations'
FLAGS.hidden1 =  10 # 'Number of units in hidden layer 1.'
FLAGS.hidden2 =  5 # 'Number of units in hidden layer 2.'
FLAGS.actfn =  'tanh' # 'activation function'
FLAGS.regNorm =  'L2' # 'regularization norm function'
FLAGS.poly_roots =  7 # 'Number of polynomial roots.'
FLAGS.reg = 1.0
FLAGS.reg_hidden1 =  FLAGS.reg # "regularization term for hidden layer 1"
FLAGS.reg_hidden2 =  FLAGS.reg # "regularization term for hidden layer 2"
FLAGS.reg_output =  FLAGS.reg # "regularization term for linear output"
FLAGS.train_dir =  'train_dir' # "directory to write training data"
#FLAGS.output_file = 'training.h5'

def main():
    # get training/test data
    polyData = tf_utils.getPolyData(FLAGS.poly_roots)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        mainAfterWithGraph(polyData)

def mainAfterWithGraph(polyData):
    # Generate placeholders for the input and labels
    x_input = tf.placeholder(tf.float32, shape=(None, 1))
    y_output = tf.placeholder(tf.float32, shape=(None, 1))

    # Build a Graph that computes predictions from the inference model.
    nnetModel, weightsPenaltyTerms = ffit.inference(x_input, 
                                                    FLAGS.hidden1, FLAGS.hidden2, 
                                                    FLAGS.actfn, FLAGS.regNorm)

    # Add to the Graph the Ops for loss calculation.
    modelError, modelLoss = ffit.loss(nnetModel, weightsPenaltyTerms,
                                      [FLAGS.reg_hidden1, FLAGS.reg_hidden2, FLAGS.reg_output],
                                      y_output)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = ffit.training(modelLoss, FLAGS.learning_rate)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()

    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    feed_dict = {x_input: tf_utils.vec2columnMat(polyData.x_train),
                 y_output:tf_utils.vec2columnMat(polyData.y_train)}

    # And then after everything is built, start the training loop.
#    saver.restore(sess, FLAGS.train_dir)
    for step in xrange(FLAGS.max_steps):
        mainStep(step, x_input, y_output, nnetModel, weightsPenaltyTerms, modelError, modelLoss,
                 train_op, summary_op, saver, sess, summary_writer, feed_dict, polyData)


def mainStep(step, x_input, y_output, nnetModel, weightsPenaltyTerms, modelError, modelLoss,
             train_op, summary_op, saver, sess, summary_writer, feed_dict, polyData):

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, loss_value = sess.run([train_op, modelLoss],
                             feed_dict=feed_dict)


    # Write the summaries and print an overview fairly often.
    if step % FLAGS.eval_steps == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f' % (step, loss_value,))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

    # Save a checkpoint and evaluate the model periodically.
    if ((step + 1) %  FLAGS.eval_steps == 0) or ((step + 1) == FLAGS.max_steps):
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        meanSqRtErrorTrain, meanSqRtErrorTest, y_all = \
                    ffit.evaluation(sess, nnetModel, x_input, polyData)
        print('Training Data Eval: train=%.2f test=%.2f' % (meanSqRtErrorTrain, 
                                                            meanSqRtErrorTest))
        


if __name__ == '__main__':
    main()
