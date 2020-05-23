import numpy as np
import tensorflow as tf

from collections import namedtuple

from lead_binary import suit_index_lookup
from binary_righty import binary_hand, get_card_index, encode_card, wins_trick_index


State = namedtuple('State', ['c', 'h'])


CARDS = [(suit + rank) for suit in 'SHDC' for rank in 'AKQJT98x']

class Player:

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

        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')
        out_card_target = graph.get_tensor_by_name('out_card_target:0')

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
        out_card = graph.get_tensor_by_name('out_card:0')

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
            cards = self.sess.run(out_card, feed_dict=feed_dict)
            next_state = (
                State(c=self.sess.run(next_c_0, feed_dict=feed_dict), h=self.sess.run(next_h_0, feed_dict=feed_dict)),
                State(c=self.sess.run(next_c_1, feed_dict=feed_dict), h=self.sess.run(next_h_1, feed_dict=feed_dict)),
                State(c=self.sess.run(next_c_2, feed_dict=feed_dict), h=self.sess.run(next_h_2, feed_dict=feed_dict)),
            )
            return cards, next_state
        return pred_fun

    def get_next_card_np(self, state, player_hand, dummy_hand, contract, last_trick_lead_i, last_trick, this_trick):
        level, strain = int(contract[0]), contract[1]

        # create input
        x = np.zeros((1, 298), np.float16)
    
        x[0, 292] = level
        if strain == 'N':
            x[0, 293] = 1
        else:
            x[0, 294 + suit_index_lookup[strain]] = 1
        x[0, 288 + last_trick_lead_i] = 1
        
        x[0, 0:32] = player_hand
        x[0, 32:64] = dummy_hand
        
        x[0, 64:96] = encode_card(last_trick[0])
        x[0, 96:128] = encode_card(last_trick[1])
        x[0, 128:160] = encode_card(last_trick[2])
        x[0, 160:192] = encode_card(last_trick[3])
        
        x[0, 192:224] = encode_card(this_trick[0])
        x[0, 224:256] = encode_card(this_trick[1])
        x[0, 256:288] = encode_card(this_trick[2])

        # get softmax output
        card, next_state = self.model(x, state)

        return card, next_state


class Cardplay:

    def __init__(self, contract, deal_str, opening_lead, players):
        self.contract = contract
        self.deal_str = deal_str
        self.opening_lead = opening_lead
        self.players = players

    def get_tricks(self):
        hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), self.deal_str.split()))
        hands_bin = list(map(binary_hand, hands))
        states = [pl.zero_state for pl in self.players]

        tricks = []
        prev_trick = ['>>', '>>', '>>', '>>']
        prev_trick_lead_index = 0

        # make the opening lead again to advance the leader's state
        card, next_state = self.players[0].get_next_card_np(
            states[0], hands_bin[0], hands_bin[1], self.contract, 0, prev_trick, ['>>', '>>', '>>'])
        states[0] = next_state
        #print('lead', CARDS[np.argmax(card)])
        opening_lead_suit, opening_lead_rank = self.opening_lead[0], self.opening_lead[1]
        opening_lead_suit_i = suit_index_lookup[opening_lead_suit]
        opening_lead_x = CARDS.index(self.opening_lead) if self.opening_lead[1] > '7' else CARDS.index(self.opening_lead[0] + 'x')
        hands_bin[0][opening_lead_x] -= 1
        hands[0][opening_lead_suit_i] = [c for c in hands[0][opening_lead_suit_i] if c != opening_lead_rank]
        assert(np.min(hands_bin[0]) == 0)

        lead_index = 0
        current_trick = ['>>', '>>', '>>', self.opening_lead]
        for _ in range(13):
            turn_i = lead_index if tricks else 1

            for _ in range(4 if tricks else 3):
                print('current trick={}, player {} on play'.format(current_trick, turn_i))

                player = self.players[turn_i]
                player_hand_bin = hands_bin[turn_i]

                assert(np.min(player_hand_bin) == 0)

                if turn_i == 0:
                    dummy_hand_bin = hands_bin[(turn_i + 1) % 4]
                elif turn_i == 2:
                    dummy_hand_bin = hands_bin[(turn_i - 1) % 4]
                else:
                    dummy_hand_bin = hands_bin[(turn_i + 2) % 4]

                card_softmax, next_state = player.get_next_card_np(
                    states[turn_i],
                    player_hand_bin,
                    dummy_hand_bin,
                    self.contract,
                    prev_trick_lead_index,
                    prev_trick,
                    current_trick[-3:]
                )
                
                p_card_i = sorted([(p, j) for j, p in enumerate(card_softmax[0])], reverse=True)
                card_p = [(CARDS[j].replace('x', '2'), p) for p, j in p_card_i if p > 0.05]
                for c, p in card_p:
                    print('%s %2.2f' % (c, p))


                states[turn_i] = next_state

                card_i = np.argmax(card_softmax)
                hands_bin[turn_i][card_i] -= 1
                assert(np.min(hands_bin[turn_i]) == 0)
                card_str = CARDS[card_i]
                if card_str.endswith('x'):
                    suit_i = suit_index_lookup[card_str[0]]
                    rank = min([c for c in hands[turn_i][suit_i] if c < '8'])
                    hands[turn_i][suit_i] = [c for c in hands[turn_i][suit_i] if c != rank]
                    card_str = card_str[0] + rank

                current_trick.append(card_str)
                del current_trick[0]

                turn_i = (turn_i + 1) % 4

            assert(len(current_trick) == 4)

            if lead_index == 0:
                trick = current_trick
            elif lead_index == 1:
                trick = current_trick[3:] + current_trick[:3]
            elif lead_index == 2:
                trick = current_trick[2:] + current_trick[:2]
            elif lead_index == 3:
                trick = current_trick[1:] + current_trick[:1]

            win_i = wins_trick_index(current_trick, self.contract[1], lead_index)

            tricks.append((lead_index, trick))
            prev_trick_lead_index = lead_index
            lead_index = win_i
            prev_trick = trick
            current_trick = ['>>', '>>', '>>', '>>']

        # assert that all cards were played
        for i in range(4):
            assert(np.min(hands_bin[i]) == 0)
            assert(np.max(hands_bin[i]) == 0)

        return tricks


def jack_auction_to_bbo(auction_str):
    return auction_str.replace('PP', 'P').replace('DD', 'D').replace('RR', 'R').lower()

def get_bbo_format(cardplay, dealer, auction):
    params = ['d=' + dealer, 'a=' + auction]
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), cardplay.deal_str.split()))
    for hand, seat in zip(hands, ['w', 'n', 'e', 's']):
        bbo_hand_str = ''
        for i, suit in enumerate(['s', 'h', 'd', 'c']):
            bbo_hand_str += suit + ''.join(hand[i]).lower()
        params.append('%s=%s' % (seat, bbo_hand_str))
    tricks = cardplay.get_tricks()
    bbo_play_str = ''
    for lead_index, trick in tricks:
        rot_trick = trick[lead_index:] + trick[:lead_index]
        bbo_play_str += ''.join(rot_trick).lower()
    params.append('p=' + bbo_play_str)

    return 'http://www.bridgebase.com/tools/handviewer.html?' + '&'.join(params)


if __name__ == '__main__':
    #KQ4.T942.543.JT2 J97.A87.2.AQ8754 A53.QJ.AQJT76.K9 T862.K653.K98.63  3H D3

    lefty = Player('lefty', './lefty_model/lefty-1000000')
    dummy = Player('dummy', './dummy_model/dummy-920000')
    righty = Player('righty', './righty_model/righty-1000000')
    declarer = Player('declarer', './decl_model/decl-1000000')

    players = [lefty, dummy, righty, declarer]

# '''
# W:7.AQ97.AQ9854.54 AQJT.KJ82.72.932 9632.54.3.KQJT76 K854.T63.KJT6.A8 #:0007 C:BW5C T:1 D:S V:ALL A:PP1DDD1S2NPPPPPP R1:2N.=.S R2:2N.+1.S.
# W:98.KJ74.QT874.K6 AK53.6532.52.Q32 64.QT9.KJ63.A974 QJT72.A8.A9.JT85 #:0006 C:BW5C T:1 D:E V:EW A:PP1SPP3SPP4SPPPPPP R1:4S.-1.S R2:4S.-1.S
# W:Q942.K762.AT52.7 T8.A84.64.AKJT62 KJ653.QJ9.Q8.984 A7.T53.KJ973.Q53 #:0008 C:BW5C T:1 D:W V:- A:PP1C1S2N3S3NPPPPPP R1:3N.-1.S R2:3N.-1.S
# W:K2.Q7.K542.AQ962 A53.T4.AT98.KJT3 Q4.A962.QJ763.54 JT9876.KJ853..87 #:0014 C:BW5C T:1 D:E V:- A:PPPP1CPP1D1S3D3S4D4SPPPPPP R1:4S.=.S R2:4S.=.S
# W:J2.A876.AK32.AQ2 KT86.QJ94.QJ9.95 Q974.T52.T75.K84 A53.K3.864.JT763 #:0013 C:BW5C T:1 D:N V:ALL A:PPPPPP1DPPPP2CPPPPPP R1:2C.-1.S R2:2C.-1.S
# W:K4.K9865.JT82.Q3 QT62.J2.K9653.A7 973.AQ4.A7.KJ964 AJ85.T73.Q4.T852 #:0019 C:BW5C T:1 D:S V:EW A:PPPPPP1CPP1HPP1NPP2DPP2HPPPP2SPPPPPP R1:2S.=.N R2:2S.=.N
# W:AQ9865..92.QT942 KT2.K8753.JT87.5 .T42.AKQ543.K873 J743.AQJ96.6.AJ6 #:0060 C:BW5C T:1 D:W V:NS A:PPPP1D1H1S3HPP4H4SPPPP5HPPPPPP R1:5H.-1.S R2:5H.=.S
# W:J84.AQ2.J876.Q54 6.8643.93.AJ9872 A97532.T975.T5.T KQT.KJ.AKQ42.K63 #:0065 C:BW5C T:1 D:N V:- A:PPPP2NPP3CPP3DPP3NPPPPPP R1:3N.-1.S R2:3N.=.S
# W:72.73.J984.KT653 AK843.AQ5.K3.A97 QJT95.86.Q65.Q82 6.KJT942.AT72.J4 #:0067 C:BW5C T:1 D:S V:EW A:2HPP2NPP3SPP4CPP4DPP4NPP5HPP5SPP5NPP6HPPPPPP R1:6H.+1.S R2:6H.+1.S
# W:T85.KT6.KQ.JT984 976.Q974.J87.A62 KJ432.832.96.KQ3 AQ.AJ5.AT5432.75 #:0071 C:BW5C T:2 D:S V:ALL A:1DPP1H1SDD2SPPPP3DPPPPPP R1:3D.+1.S R2:3D.+1.S

# W:43.J98.KQ95.AQJ4 AKT975.KQT6.72.K QJ862.7532.J6.73 .A4.AT843.T98652 #:0093 C:BW5C T:1 D:N V:ALL A:1SPP1NPP2HPP3CPPPPPP R1:3C.=.S R2:3C.=.S
# '''

    data = [
        
        ('S', '2N', 'D8', 'PP1DDD1S2NPPPPPP', '7.AQ97.AQ9854.54 AQJT.KJ82.72.932 9632.54.3.KQJT76 K854.T63.KJT6.A8'),
        ('E', '4S', 'D4', 'PP1SPP3SPP4SPPPPPP', '98.KJ74.QT874.K6 AK53.6532.52.Q32 64.QT9.KJ63.A974 QJT72.A8.A9.JT85'),
        ('W', '3N', 'S2', 'PP1C1S2N3S3NPPPPPP', 'Q942.K762.AT52.7 T8.A84.64.AKJT62 KJ653.QJ9.Q8.984 A7.T53.KJ973.Q53'),
        ('E', '4S', 'D2', 'PPPP1CPP1D1S3D3S4D4SPPPPPP', 'K2.Q7.K542.AQ962 A53.T4.AT98.KJT3 Q4.A962.QJ763.54 JT9876.KJ853..87'),
        ('N', '2C', 'DA', 'PPPPPP1DPPPP2CPPPPPP', 'J2.A876.AK32.AQ2 KT86.QJ94.QJ9.95 Q974.T52.T75.K84 A53.K3.864.JT763'),
        ('W', '5H', 'D9', 'PPPP1D1H1S3HPP4H4SPPPP5HPPPPPP', 'AQ9865..92.QT942 KT2.K8753.JT87.5 .T42.AKQ543.K873 J743.AQJ96.6.AJ6'),
        ('N', '3N', 'S4', 'PPPP2NPP3CPP3DPP3NPPPPPP', 'J84.AQ2.J876.Q54 6.8643.93.AJ9872 A97532.T975.T5.T KQT.KJ.AKQ42.K63'),
        ('S', '6H', 'S7', '2HPP2NPP3SPP4CPP4DPP4NPP5HPP5SPP5NPP6HPPPPPP', '72.73.J984.KT653 AK843.AQ5.K3.A97 QJT95.86.Q65.Q82 6.KJT942.AT72.J4'),
        ('S', '3D', 'S5', '1DPP1H1SDD2SPPPP3DPPPPPP', 'T85.KT6.KQ.JT984 976.Q974.J87.A62 KJ432.832.96.KQ3 AQ.AJ5.AT5432.75'),
        ('N', '3C', 'C4', '1SPP1NPP2HPP3CPPPPPP', '43.J98.KQ95.AQJ4 AKT975.KQT6.72.K QJ862.7532.J6.73 .A4.AT843.T98652'),

        ('W', '3S', 'DA', '2DPP3DPPPPDDPP3SPPPPPP', '73.85.AJT975.532 KJ8.J964.2.AJT87 Q92.AT7.KQ64.K94 AT654.KQ32.83.Q6'),
    ]

    for dealer, contract, opening_lead, auction, deal_str in data:
        cardplay = Cardplay(contract, deal_str, opening_lead, players)
        print(contract)
        print(deal_str)
        # for trick in cardplay.get_tricks():
        #     print(trick)
        #print(get_bbo_format(cardplay, dealer, jack_auction_to_bbo(auction)))
        cardplay.get_tricks()
