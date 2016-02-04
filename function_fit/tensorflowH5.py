from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py

def _getShape(npArray):
    try:
        shape = npArray.shape
    except AttributeError:
        shape = ()
    return shape

def _makeList(listOrSingle):
    if isinstance(listOrSingle, list) or isinstance(listOrSingle, tuple):
        return listOrSingle
    return [listOrSingle]

def _getTensorNamesToUse(names, tensors):
    if names is None:
        return [tensor.name for tensor in tensors]
    if isinstance(names, str):
        assert len(tensors)==1
        return [names]
    assert isinstance(names, list) or isinstance(names, tuple)
    assert len(names)==len(tensors)
    assert len(set(names)) == len(names)
    return names

def _calcChunkSize(shape, dtype):
    numElems = 1
    for sz in shape:
        numElems *= sz
    npDtype2itemSize = {np.int32:4, np.int64:8, np.int16:2, np.int8:1,
                        np.uint32:4, np.uint64:8, np.uint16:2, np.uint8:1,
                        np.float32:4, np.float64:8}
    
    bytesPerElem = npDtype2itemSize.get(dtype,1)
    bytesPerTensor = bytesPerElem * numElems
    maxChunkSize = max(1,(16<<20)//bytesPerTensor)
    chunkSize = min(500, maxChunkSize)
    return chunkSize

def _appendStepAndData(group, step, npArray):
    stepDs = group['step']
    dataDs = group['data']
    stepLen = len(stepDs)
    dataLen = len(dataDs)
    assert stepLen == dataLen
    stepDs.resize((stepLen+1,))
    shape = _getShape(npArray)
    dataDs.resize((stepLen+1,)+shape)
    stepDs[stepLen] = step
    if len(shape)==0:
        dataDs[stepLen] = npArray
    else:
        dataDs[stepLen,:] = npArray[:]


###################################################    
class TensorFlowH5Writer(object):
    def __init__(self, fname):
        self.fname = fname
        self.h5 = h5py.File(fname, 'w')

    def addTensors(self, sess, step, tensors, names=None, feed_dict=None):
        tensors = _makeList(tensors)
        names = _getTensorNamesToUse(names, tensors)
#        shapes = map(tuple,sess.run(map(tf.shape, tensors), feed_dict=feed_dict))
        dtypes = [tensor.dtype.as_numpy_dtype for tensor in tensors]
        npArrays = sess.run(tensors, feed_dict=feed_dict)
        for name, dtype, npArray in zip(names, dtypes, npArrays):
            gr = self.getGroup(name, _getShape(npArray), dtype)
            _appendStepAndData(gr, step, npArray)

    def addNumpyArrays(self, step, numpyArrays, names):
        numpyArrays = _makeList(numpyArrays)
        names = _makeList(names)
        assert len(names)==len(numpyArrays)
        for npArray, name in zip(numpyArrays, names):
            dtype = npArray.dtype
            gr = self.getGroup(name, _getShape(npArray), dtype)
            _appendStepAndData(gr, step, npArray)

    def getGroup(self, name, shape, dtype):
        if name in self.h5.keys():
            gr = self.h5[name]
        else:
            try:
                gr = self.h5.create_group(name)
            except ValueError:
                import IPython
                IPython.embed()
                
            gr.create_dataset('step', (0,), dtype='i4', 
                              chunks=(1000,), maxshape=(None,))
            dsetShape = (0,) + shape
            chunkSize = _calcChunkSize(shape, dtype)
            chunkShape = (chunkSize,) + shape
            gr.create_dataset('data', dsetShape, dtype=dtype, 
                              chunks=chunkShape, maxshape=(None,) + shape)
        return gr

###################################################    
class TensorFlowH5Plotter(object):
    def __init__(self, fname, plt=None):
        self.fname = fname
        self.h5 = h5py.File(fname, 'r')
        if plt is None:
            import matplotlib.pyplot as plt
        self.plt = plt
        
    def scalarSummary(self, names):
        for name in names:
            step = self.h5[name]['step'][:]
            data = self.h5[name]['data'][:]
            self.plt.plot(step, data, label=name)
        self.plt.xlabel('step')
        self.plt.legend()

    def singleInputWeightBiases(self, weights, biases, steps='latest'):
        assert steps == 'latest'
        weightSteps = self.h5[weights]['step'][:]
        lastWeights = self.h5[weights]['data'][len(weightSteps)-1,:].flatten()
        lastBiases = self.h5[biases]['data'][len(weightSteps)-1,:]
        self.plt.plot(lastBiases, lastWeights, '*')
        self.plt.xlabel(biases)
        self.plt.ylabel(weights)
        
        
if __name__ == '__main__':
    # test
    h5writer = TensorFlowH5Writer('testfile.h5')
    matrix1 = tf.constant([[3., 3.]], name='mat1')
    matrix2 = tf.constant([[2.],[2.]], name='mat2')
    product = tf.matmul(matrix1, matrix2, name='product')
    x = tf.constant(3, name='x')
    y = tf.constant(3.3, name='y')

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
        h5writer.addTensors(step=0, sess=sess, tensors=[x,y,matrix1, matrix2, product],feed_dict={})
        h5writer.addNumpyArrays(step=0, numpyArrays=np.zeros(3), names="dude")

