import sys
import numpy as np
import tensorflow as tf

from batcher import Batcher

model_path = sys.argv[1]

batch_size = 64
n_iterations = 500000
display_step = 1000

X_train = np.load('./X_train.npy')
y_train = np.load('./y_train.npy')

X_val = np.load('./X_val.npy')
y_val = np.load('./y_val.npy')

n_examples = y_train.shape[0]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

lstm_size = 512
n_layers = 4

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)
    
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out = tf.placeholder(tf.float32, [None, None, n_bids], 'seq_out')

softmax_w = tf.get_variable('softmax_w', shape=[lstm_cell.output_size, n_bids], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_bid_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_bid_logit')
out_bid_target = tf.reshape(seq_out, [-1, n_bids], name='out_bid_target')

cost = tf.losses.softmax_cross_entropy(out_bid_target, out_bid_logit)

train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, 10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=20)

    for i in range(n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, y_train])
        if i % display_step == 0:
            x_cost, y_cost = cost_batch.next_batch([X_train, y_train])
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            c_valid = sess.run(cost, feed_dict={seq_in: X_val, seq_out: y_val, keep_prob: 1.0})
            print('{}. c_train={} c_valid={}'.format(i, c_train, c_valid))
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.6})

    saver.save(sess, model_path, global_step=n_iterations)
