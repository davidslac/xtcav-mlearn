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
    def __init__(self):
        pass

    def h5write(self, openH5file, groupName='FLAGS'):
        gr = openH5file.create_group(groupName)
        for attr in dir(self):
            if attr.startswith('_'): continue
            try:
                gr.create_dataset(attr, data=getattr(self, attr))
            except:
                pass


FLAGS = Bag()
FLAGS.learning_rate =  0.005 # "initial learning rate"
FLAGS.max_steps =  5000 # 'Numer of steps to run trainer'
FLAGS.num_eval_updates =  20 # "Number of evaluations to do during training"
FLAGS.eval_steps =  FLAGS.max_steps//FLAGS.num_eval_updates # 'steps between evaluations'
FLAGS.hidden1 =  200 # 'Number of units in hidden layer 1.'
FLAGS.hidden2 =  200 # 'Number of units in hidden layer 2.'
FLAGS.actfn =  'sigmoid' # 'activation function - one of tanh, sigmoid, softplus relu relu6
FLAGS.regNorm =  'L2' # 'regularization norm function'
FLAGS.poly_roots =  7 # 'Number of polynomial roots.'
FLAGS.reg = 0.01
FLAGS.reg_hidden1 =  FLAGS.reg # "regularization term for hidden layer 1"
FLAGS.reg_hidden2 =  FLAGS.reg # "regularization term for hidden layer 2"
FLAGS.reg_output =  0.0 #FLAGS.reg # "regularization term for linear output"
FLAGS.train_dir =  'train_dir' # "directory to write training data"
FLAGS.summaryfile = 'train_dir/run_summary.h5'
FLAGS.numberCores = 4

def main():
    # get training/test data
    functionData = tf_utils.FunctionData(poly_degree=FLAGS.poly_roots)
    mod = Bag()
                            
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        mainAfterWithGraph(functionData, mod)

def mainAfterWithGraph(functionData, mod):
    # Generate placeholders for the input and labels
    mod.x_input = tf.placeholder(tf.float32, shape=(None, 1))
    mod.y_output = tf.placeholder(tf.float32, shape=(None, 1))

    # Build a Graph that computes predictions from the inference model.
    mod = ffit.inference(mod, mod.x_input, FLAGS.hidden1, FLAGS.hidden2, 
                   FLAGS.actfn, FLAGS.regNorm)

    # Add to the Graph the Ops for loss calculation.
    mod.modelError, mod.loss = ffit.loss(mod.nnetModel, mod.W_regterms,
                                         [FLAGS.reg_hidden1, FLAGS.reg_hidden2, FLAGS.reg_output],
                                         mod.y_output)

    # Add to the Graph the Ops that calculate and apply gradients.
    mod = ffit.training(mod, mod.loss, FLAGS.learning_rate)
    
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
    FLAGS.h5write(h5out.h5)
    functionData.h5write(h5out.h5)

    train_feed_dict = {mod.x_input: functionData.x_train,
                       mod.y_output:functionData.y_train}

    test_feed_dict = {mod.x_input: functionData.x_test,
                      mod.y_output:functionData.y_test}

    all_feed_dict = {mod.x_input: functionData.x_all,
                     mod.y_output:functionData.y_all}

    # And then after everything is built, start the training loop.
#    saver.restore(sess, FLAGS.train_dir)

    mod.grad = None
    mod.gradMag = None

    for step in xrange(FLAGS.max_steps):
        trainStep(step, mod, sess, h5out, train_feed_dict, test_feed_dict, all_feed_dict)


def trainStep(step, mod, sess, h5out, train_feed_dict, test_feed_dict, all_feed_dict):
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, trainLoss = sess.run([mod.train_op, mod.loss],
                            feed_dict=train_feed_dict)

    # Write the summaries and print an overview fairly often.
    if step % FLAGS.eval_steps == 0:
        trainAvgErr = sess.run(mod.modelError, feed_dict=train_feed_dict)
        grad = tf_utils.getFlattendGrad(sess, mod.grads_and_vars, train_feed_dict)
        gradMag = np.sqrt(np.dot(grad,grad))
        if mod.grad is not None:
            cosWithPrev = np.dot(mod.grad, grad)/(mod.gradMag * gradMag)
            h5out.addNumpyArrays(step, [gradMag, cosWithPrev], ['grad_mag', 'grad_cos'])
        mod.grad = grad
        mod.gradMag = gradMag

        testAvgErr = sess.run(mod.modelError, feed_dict=test_feed_dict)
        h5out.addNumpyArrays(step, [trainAvgErr, testAvgErr, trainLoss],
                             ['trainAvgErr', 'testAvgErr', 'trainLoss'])

        trainFunction = sess.run(mod.nnetModel, feed_dict=all_feed_dict)
        h5out.addNumpyArrays(step, trainFunction, ['trainFunction'])

        h5out.addTensors(sess, step, [mod.H1_W, 
                                      mod.H1_B, 
                                      mod.H2_W,
                                      mod.H2_B,
                                      mod.L_W, 
                                      mod.L_B],
                                     ['H1_W', 
                                      'H1_B', 
                                      'H2_W',
                                      'H2_B',
                                      'L_W', 
                                      'L_B'])
        # Print status to stdout.
        print('Step %d: loss = %.2f trainErr=%.2f testErr=%.2f' % (step, trainLoss, trainAvgErr, testAvgErr))
        # Update the events file.
#        summary_str = sess.run(summary_op, feed_dict=feed_dict)
#        summary_writer.add_summary(summary_str, step)

 
def readFlags(h5, groupName='FLAGS'):
    if groupName not in h5.keys():
        return ''
    msg = ''
    gr = h5[groupName]
    flagsPerLine = 5
    keys = gr.keys()
    keys.sort()
    msgItems = []
    for ii,dset in enumerate(keys):
        try:
            msgItems.append('%s=%s' % (dset, gr[dset][:]))
        except:
            msgItems.append('%s=%s' % (dset, gr[dset].value))
    return '\n'.join(msgItems)

def plotSummaryFile(fname):
    import matplotlib.pyplot as plt
    h5plt = tfh5.TensorFlowH5(fname, plt)
    functionData = tf_utils.FunctionData(h5=h5plt.h5)

    print("==FLAGS==\n%s" % readFlags(h5plt.h5))
    plt.figure(10,figsize=(18,12))    

    plt.subplot(2,3,1)
    h5plt.scalarSummary(['trainAvgErr', 'testAvgErr', 'trainLoss'])
    plt.title('train/test/loss')

    plt.subplot(2,3,2)
    h5plt.singleInputWeightBiases('H1_W', 'H1_B')
    plt.title('scatter plot of hidden1 weights/biases')

    plt.subplot(2,3,3)
    h5plt.histogram(['H1_W', 'H1_B',
                     'H2_W', 'H2_B',
                     'L_W','L_B'])
    plt.title('scatter plot of weights/outputs/biases')
    
    plt.subplot(2,3,4)
    h5plt.scalarSummary(['grad_mag'])
    plt.title('magnitude of gradient')

    plt.subplot(2,3,5)
    h5plt.scalarSummary(['grad_cos'])
    plt.title('cosine of gradient between step interval')

    plt.subplot(2,3,5)
    h5plt.scalarSummary(['grad_cos'])
    plt.title('cosine of gradient between step interval')

    plt.subplot(2,3,6)
    h5plt.curveFamily(functionData.x_all, 'trainFunction')
    functionData.plot(plt)
    plt.title('function data vs train')

    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
    plotSummaryFile(FLAGS.summaryfile)
