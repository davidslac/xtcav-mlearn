import numpy as np
import tensorflow as tf

def main():
    x_input = tf.placeholder(tf.float32, shape=(None, 1))
    y_output = tf.placeholder(tf.float32, shape=(None, 1))

    hidden_weights = tf.Variable(tf.truncated_normal([1,10], stddev=0.1), name='weights')
    output_weights = tf.Variable(tf.truncated_normal([10,1], stddev=0.1), name='output')
    inference = tf.matmul(tf.matmul(x_input, hidden_weights), output_weights)
    regA = tf.reduce_sum(tf.pow(hidden_weights, 2))
    regB = tf.reduce_sum(tf.pow(output_weights, 2))
    modelError = tf.reduce_mean(tf.pow(tf.sub(inference,y_output),2), name='model-error')

    fail = True
    if fail:
        regularizationTerm = tf.Variable(tf.zeros([], dtype=np.float32), name='regterm')
        regularizationTerm +=  tf.mul(2.0, regA)
        regularizationTerm +=  tf.mul(3.0, regB)
    else:
        regularizationTerm = tf.add(tf.mul(2.0, regA), tf.mul(3.0, regB), name='regterm')

    loss = tf.add(modelError, regularizationTerm, name='loss')
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter('train_dir', 
                                            graph_def=sess.graph_def)

    feed_dict = {x_input:np.ones((30,1), dtype=np.float32),
                 y_output:np.ones((30,1), dtype=np.float32)}

    for step in xrange(1000):
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        if step % 100 == 0:
            print( "step=%d loss=%.2f" % (step, loss_value))
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

if __name__ == '__main__':
    main()
