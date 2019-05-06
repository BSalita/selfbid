import sys
import numpy as np
import tensorflow as tf

from batcher import Batcher

model_path = './decl_model/decl'

batch_size = 64
n_iterations = 1000000
display_step = 10000


X_train = np.load('./decl_bin/X_train.npy')
Y_train = np.load('./decl_bin/Y_train.npy')

X_val = np.load('./decl_bin/X_val.npy')
Y_val = np.load('./decl_bin/Y_val.npy')

n_examples = Y_train.shape[0]
n_ftrs = X_train.shape[2]
n_cards = Y_train.shape[2]

lstm_size = 128
n_layers = 3

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)

state = []
for i, cell_i in enumerate(cells):
    s_c = tf.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
    s_h = tf.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
    state.append(tf.contrib.rnn.LSTMStateTuple(c=s_c, h=s_h))
state = tuple(state)

x_in = tf.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out = tf.placeholder(tf.float32, [None, None, n_cards], 'seq_out')

softmax_w = tf.get_variable('softmax_w', shape=[lstm_cell.output_size, n_cards], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_card_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_card_logit')
out_card_target = tf.reshape(seq_out, [-1, n_cards], name='out_card_target')

output, next_state = lstm_cell(x_in, state)

out_card = tf.nn.softmax(tf.matmul(output, softmax_w), name='out_card')

for i, next_i in enumerate(next_state):
    tf.identity(next_i.c, name='next_c_{}'.format(i))
    tf.identity(next_i.h, name='next_h_{}'.format(i))

cost = tf.losses.softmax_cross_entropy(out_card_target, out_card_logit)

train_step = tf.train.AdamOptimizer(0.0003).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, 10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=50)

    for i in range(n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, Y_train])
        if i % display_step == 0:
            print(i)
            x_cost, y_cost = cost_batch.next_batch([X_train, Y_train])
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            c_valid = sess.run(cost, feed_dict={seq_in: X_val, seq_out: Y_val, keep_prob: 1.0})
            print('{}. c_train={} c_valid={}'.format(i, c_train, c_valid))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)
