import numpy as np
import tensorflow as tf
import bidding

from collections import namedtuple

from binary_sayc import DealData
from simulator import Simulator


State = namedtuple('State', ['c', 'h'])


class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.sess = tf.InteractiveSession()
        self.load_model()
        self.lstm_size = 128
        self.zero_state = (
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
        )
        self.nesw_initial = [self.zero_state, self.zero_state, self.zero_state, self.zero_state]
        self.model = self.init_model()
        
    def close(self):
        self.sess.close()
        
    def load_model(self):
        saver = tf.train.import_meta_graph(self.model_path + '.meta')
        saver.restore(self.sess, self.model_path)
        
    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')
        seq_out = graph.get_tensor_by_name('seq_out:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        out_bid_logit = graph.get_tensor_by_name('out_bid_logit:0')
        out_bid_target = graph.get_tensor_by_name('out_bid_target:0')

        state_c_0 = graph.get_tensor_by_name('state_c_0:0')
        state_h_0 = graph.get_tensor_by_name('state_h_0:0')

        state_c_1 = graph.get_tensor_by_name('state_c_1:0')
        state_h_1 = graph.get_tensor_by_name('state_h_1:0')

        state_c_2 = graph.get_tensor_by_name('state_c_2:0')
        state_h_2 = graph.get_tensor_by_name('state_h_2:0')

        next_c_0 = graph.get_tensor_by_name('next_c_0:0')
        next_h_0 = graph.get_tensor_by_name('next_h_0:0')

        next_c_1 = graph.get_tensor_by_name('next_c_1:0')
        next_h_1 = graph.get_tensor_by_name('next_h_1:0')

        next_c_2 = graph.get_tensor_by_name('next_c_2:0')
        next_h_2 = graph.get_tensor_by_name('next_h_2:0')

        x_in = graph.get_tensor_by_name('x_in:0')
        out_bid = graph.get_tensor_by_name('out_bid:0')
        
        # defining model
        p_keep = 1.0
        
        def pred_fun(x, state_in):
            feed_dict = {
                keep_prob: p_keep,
                x_in: x,
                state_c_0: state_in[0].c,
                state_h_0: state_in[0].h,
                state_c_1: state_in[1].c,
                state_h_1: state_in[1].h,
                state_c_2: state_in[2].c,
                state_h_2: state_in[2].h,
            }
            bids = self.sess.run(out_bid, feed_dict=feed_dict)
            next_state = (
                State(c=self.sess.run(next_c_0, feed_dict=feed_dict), h=self.sess.run(next_h_0, feed_dict=feed_dict)),
                State(c=self.sess.run(next_c_1, feed_dict=feed_dict), h=self.sess.run(next_h_1, feed_dict=feed_dict)),
                State(c=self.sess.run(next_c_2, feed_dict=feed_dict), h=self.sess.run(next_h_2, feed_dict=feed_dict)),
            )
            return bids, next_state
        return pred_fun
        
    def simulate_auction(self, deal_data):
        deal_data.reset_auction()
        sim = Simulator(deal_data, {}, self.model)
        return sim.simulate_auction(deal_data.auction, self.nesw_initial, max_bid=True)

    def next_bid(self, deal_data, auction):
        sim = Simulator(deal_data, {}, self.model)

        turn_i = len(auction) % 4  # index of whose turn it is

        state = self.zero_state

        for bid_i, bid in enumerate(deal_data.auction):
            if bid == 'PAD_START':
                continue
            if bid_i % 4 == turn_i:
                _, next_state = sim.next_bid_np(auction[:bid_i], state)
                state = next_state

        # now the auction is replayed and it's really my turn
        out_bid_np, _ = sim.next_bid_np(auction, state)

        return bidding.bid_max_bid(auction, out_bid_np)

