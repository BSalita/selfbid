import numpy as np


## input
#   0:32  player hand
#  32:64  seen hand (dummy, or declarer if we create data for dummy)
#  64:96  last trick card 0
#  96:128 last trick card 1
# 128:160 last trick card 2
# 160:192 last trick card 3
# 192:224 this trick lho card
# 224:256 this trick pard card
# 256:288 this trick rho card
# 288:292 last trick lead player index one-hot
#     292 level
# 293:298 strain one hot N, S, H, D, C
class BinaryInput:

    def __init__(self, x):
        self.x = x
        self.n_samples, self.n_ftrs = x.shape

    def set_player_hand(self, hand_bin):
        self.x[:, :32] = hand_bin

    def get_player_hand(self):
        return self.x[:, :32]

    def set_public_hand(self, hand_bin):
        self.x[:, 32:64] = hand_bin 

    def get_public_hand(self):
        # public hand is usually dummy (unless the player is dummy, then public is declarer)
        return self.x[:, 32:64]

    def set_last_trick(self, last_trick):
        self.x[:, 64:192] = last_trick.reshape((self.n_samples, 4*32))

    def get_last_trick(self):
        return self.x[:, 64:192].reshape((self.n_samples, 4, 32))

    def set_this_trick(self, this_trick):
        self.x[:, 192:288] = this_trick.reshape((self.n_samples, 3*32))

    def get_this_trick(self):
        return self.x[:, 192:288].reshape((self.n_samples, 3, 32))

    def set_last_trick_lead(self, last_trick_lead_i):
        self.x[:, 288:292] = 0
        #self.x[]
        self.x[last_trick_lead_i == 0, 288] = 1
        self.x[last_trick_lead_i == 1, 289] = 1
        self.x[last_trick_lead_i == 2, 290] = 1
        self.x[last_trick_lead_i == 3, 291] = 1

    def get_last_trick_lead(self):
        return np.argmax(self.x[:, 288:292], axis=1)

    def set_level(self, level):
        self.x[:, 292] = level

    def get_level(self):
        return self.x[:, 292]

    def set_strain(self, strain):
        self.x[:, 293:298] = strain

    def get_strain(self):
        return self.x[:, 293:298]


def get_cards_from_binary_hand(hand):
    cards = []
    for i, count in enumerate(hand):
        for _ in range(int(count)):
            cards.append(i)
    return np.array(cards)

def get_binary_hand_from_cards(cards):
    hand = np.zeros(32)
    for card in cards:
        hand[int(card)] += 1
    return hand
