import sys
import datetime
import numpy as np
import tensorflow as tf
import bidding

from batcher import Batcher

model_path = './lead_model_jack/lead_jack'

seed = 1337

batch_size = 64
n_iterations = 5000000
display_step = 20000

A_train = np.load('../data/leads_bin2/A_train.npy')
H_train = np.load('../data/leads_bin2/H_train.npy')
C_train = np.load('../data/leads_bin2/C_train.npy')
L_train = np.load('../data/leads_bin2/L_train.npy')

A_val = np.load('../data/leads_bin2/A_val.npy')
H_val = np.load('../data/leads_bin2/H_val.npy')
C_val = np.load('../data/leads_bin2/C_val.npy')
L_val = np.load('../data/leads_bin2/L_val.npy')


n_examples = A_train.shape[0]
n_bids = A_train.shape[1]
n_bid_ftrs = A_train.shape[2]
n_hand_ftrs = H_train.shape[1]

n_hidden_units = [512, 128]
#n_hidden_units = [1024, 256, 256]

lstm_size = 64
n_layers = 2

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
l2_reg = 0.04

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

lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_bid_ftrs], 'seq_in')

#batch_size_tf = tf.placeholder(tf.int32, [], 'batch_size_tf')

# initial_state = lstm_cell.zero_state(batch_size=batch_size_tf, dtype=tf.float32)

# print(type(initial_state))
# print(len(initial_state))
# print(initial_state)

#out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, initial_state=initial_state, dtype=tf.float32)
out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)
print(out_rnn)
print(out_rnn.shape)

H = tf.placeholder(tf.float32, shape=[None, n_hand_ftrs], name='H')
C = tf.placeholder(tf.float32, shape=[None, 52], name='H')
L = tf.placeholder(tf.float32, shape=[None, 52], name='L')

fc_in = tf.concat([out_rnn[:, -1, :], H], axis=1, name='fc_in')

print(fc_in)

fc_w = tf.get_variable('fcw', shape=[fc_in.shape.as_list()[1], n_hidden_units[0]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z = tf.matmul(fc_in, fc_w)
fc_a = tf.nn.relu(fc_z)

fc_w_1 = tf.get_variable('fcw_1', shape=[fc_a.shape.as_list()[1], n_hidden_units[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z_1 = tf.matmul(fc_a, fc_w_1)
fc_a_1 = tf.nn.relu(fc_z_1)

w_out = tf.get_variable('w_out', shape=[n_hidden_units[1], 52], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))

#pred = tf.nn.softmax(tf.matmul(fc_a_1, w_out), name='pred')
pred_logit = tf.matmul(fc_a_1, w_out, name='pred_logit')
#pred = tf.multiply(H[:, 18:], tf.nn.softmax(pred_logit), name='pred')
pred = tf.nn.softmax(pred_logit, name='pred')

weights = [fc_w, fc_w_1, w_out]



#cost_pred = tf.reduce_sum(tf.squared_difference(pred, L))
#cost_pred = tf.reduce_sum(tf.multiply(pred, C))
cost_pred = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=L))
cost_reg = l2_reg * (1.0 / (2*batch_size)) * sum([tf.reduce_sum(tf.square(w)) for w in weights])

cost = cost_pred/batch_size + cost_reg

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#train_step = tf.train.MomentumOptimizer(0.0001, momentum=0.8).minimize(cost)
#train_step = tf.train.MomentumOptimizer(0.001, momentum=0.9).minimize(cost)
#train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

def metric(predictions, costs):
    predicted_indexes = np.argmax(predictions, axis=1)
    result = []
    for i in range(len(predictions)):
        result.append(costs[i, predicted_indexes[i]])
        
    return result


batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, 10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=50)

    for i in range(n_iterations):
        a_batch, h_batch, c_batch, l_batch = batch.next_batch([A_train, H_train, C_train, L_train])
        if i % display_step == 0:
            a_cost, h_cost, c_cost, l_cost = cost_batch.next_batch([A_train, H_train, C_train, L_train])
            c_train_pred = sess.run(cost_pred, feed_dict={seq_in: a_cost, H: h_cost, C: c_cost, L: l_cost, keep_prob: 1.0})
            c_train_reg = sess.run(cost_reg, feed_dict={seq_in: a_cost, H: h_cost, C: c_cost, L: l_cost, keep_prob: 1.0})
            c_valid_pred = sess.run(cost_pred, feed_dict={seq_in: A_val, H: H_val, C: C_val, L: L_val, keep_prob: 1.0})
            pred_valid = sess.run(pred, feed_dict={seq_in: A_val, H: H_val, keep_prob: 1.0})
            metric_valid = np.mean(metric(pred_valid, C_val))
            pred_train = sess.run(pred, feed_dict={seq_in: a_cost, H: h_cost, keep_prob: 1.0})
            metric_train = np.mean(metric(pred_train, c_cost))
            print('{}. c_train_reg={} c_train_pred={} c_valid_pred={} tricks_train={} tricks_valid={} t={}'.format(
                i, 
                c_train_reg, 
                c_train_pred/10000, 
                c_valid_pred/10000,
                metric_train,
                metric_valid,
                datetime.datetime.now().isoformat()))
            sys.stdout.flush()

            saver.save(sess, model_path, global_step=i)
        
        sess.run(train_step, feed_dict={seq_in: a_batch, H: h_batch, L: l_batch, keep_prob: 0.8, learning_rate: 0.0003 / (2**(i/1e6))})

    saver.save(sess, model_path, global_step=n_iterations)
