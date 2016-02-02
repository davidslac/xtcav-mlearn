from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import function_fit
import tf_utils

import unittest

class MyTest(unittest.TestCase):
  def setUp(self):
    self.x = tf.placeholder(tf.float32, shape=(None, 1))
    self.y = tf.placeholder(tf.float32, shape=(None, 1))
    self.hidden1 = 10
    self.hidden2 = 5
    self.regLambda = [1,1,1]
    self.learningRate = 0.05

  def tearDown(self):
    pass

  def test_polyData(self):
    polyData = tf_utils.getPolyData(poly_degree=3, numPoints=20 )
    print("%s" % polyData.x_all[0:3])
    print("%s" % polyData.y_all[0:3])

  def test_inference(self):
    nnet, weightTerms = function_fit.inference(self.x, self.hidden1, self.hidden2)
    self.assertEqual(len(weightTerms),3, msg='expected 3 weight terms')
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
      nnetEval, w0, w1, w2 = sess.run([nnet, weightTerms[0], weightTerms[1], weightTerms[2]], 
                                         feed_dict={self.x:np.ones((3,1), dtype=np.float32)})
      print("nnet=%r\n w0=%r\n w1=%r\n w2=%r" % (nnetEval, w0, w1, w2))

  def test_loss(self):
    nnet, weightTerms = function_fit.inference(self.x, self.hidden1, self.hidden2)
    modError, modLoss = function_fit.loss(nnet, weightTerms, self.regLambda, self.y)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
      print("loss=%r" % sess.run(modLoss, feed_dict={self.x:np.ones((3,1), dtype=np.float32), 
                                                     self.y:np.ones((3,1), dtype=np.float32)}))
    
  def test_loss(self):
    nnet, weightTerms = function_fit.inference(self.x, self.hidden1, self.hidden2)
    modError, modLoss = function_fit.loss(nnet, weightTerms, self.regLambda, self.y)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
      print("loss=%r" % sess.run(modLoss, feed_dict={self.x:np.ones((3,1), dtype=np.float32), 
                                                     self.y:np.ones((3,1), dtype=np.float32)}))
    
  def test_train(self):
    nnet, weightTerms = function_fit.inference(self.x, self.hidden1, self.hidden2)
    modError, modLoss = function_fit.loss(nnet, weightTerms, self.regLambda, self.y)
    train_op = function_fit.training(modLoss, self.learningRate)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
      print("loss=%r" % sess.run(modLoss, feed_dict={self.x:np.ones((3,1), dtype=np.float32), 
                                                     self.y:np.ones((3,1), dtype=np.float32)}))
    


if __name__ == "__main__":
    # now run the tests in unitTests, this routine does not return.
    unittest.main(argv=[sys.argv[0], '-v'])
