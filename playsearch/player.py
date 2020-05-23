import numpy as np
import tensorflow as tf


class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.sess = tf.InteractiveSession()
        self.load_model()
        self.model = self.init_model()


    def close(self):
        self.sess.close()

    def load_model(self):
        saver = tf.train.import_meta_graph(self.model_path + '.meta')
        saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')

        p_keep = 1.0

        def pred_fun(x):
            card_logit = self.sess.run(out_card_logit, feed_dict={seq_in: x, keep_prob: p_keep})
            return self.sess.run(tf.nn.softmax(card_logit.reshape((x.shape[0], x.shape[1], 32))))[:,-1,:]

        return pred_fun

    def next_cards_softmax(self, x):
        return self.model(x)

