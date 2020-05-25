import numpy as np
import tensorflow as tf


SUIT_MASK = np.array([
    [1] * 8 + [0] * 24,
    [0] * 8 + [1] * 8 + [0] * 16,
    [0] * 16 + [1] * 8 + [0] * 8,
    [0] * 24 + [1] * 8,
])


class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.load_model()
        self.model = self.init_model()

    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')  #  we always give the whole sequence from the beginning. shape = (batch_size, n_tricks, n_features)
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')  #  shows which card it would play at each trick. (but we only care about the card for last trick)

        p_keep = 1.0

        def pred_fun(x):
            result = None
            with self.graph.as_default():
                card_logit = self.sess.run(out_card_logit, feed_dict={seq_in: x, keep_prob: p_keep})
                result = self.reshape_card_logit(card_logit, x)
            return result

        return pred_fun

    def reshape_card_logit(self, card_logit, x):
        return self.sess.run(tf.nn.softmax(card_logit.reshape((x.shape[0], x.shape[1], 32))))[:,-1,:]

    def next_cards_softmax(self, x):
        return self.model(x)


class BatchPlayerLefty(BatchPlayer):

    def reshape_card_logit(self, card_logit, x):
        return self.sess.run(tf.nn.softmax(card_logit.reshape((x.shape[0], x.shape[1] - 1, 32))))[:,-1,:]


def follow_suit(cards_softmax, own_cards, trick_suit):
    assert cards_softmax.shape[1] == 32
    assert own_cards.shape[1] == 32
    assert trick_suit.shape[1] == 4
    assert trick_suit.shape[0] == cards_softmax.shape[0]
    assert cards_softmax.shape[0] == own_cards.shape[0]

    suit_defined = np.max(trick_suit, axis=1) > 0
    trick_suit_i = np.argmax(trick_suit, axis=1)

    mask = (own_cards > 0).astype(np.int)

    mask[suit_defined] *= SUIT_MASK[trick_suit_i[suit_defined]]

    legal_cards_softmax = cards_softmax * mask

    s = np.sum(legal_cards_softmax, axis=1, keepdims=True)
    s[s < 1e-9] = 1

    return legal_cards_softmax / s
