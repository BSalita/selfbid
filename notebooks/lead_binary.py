import sys
import os
import os.path
import numpy as np
import jack_to_deal_auction as jack
import bidding
from data_access import card_index_lookup


suit_index_lookup = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
seats = ['W', 'N', 'E', 'S']
seat_index = {'W': 0, 'N': 1, 'E': 2, 'S': 3}


def convert_auction(auction_str):
    return auction_str.strip().replace('PP', 'PASS').replace('DD', 'X').replace('RR', 'XX').split()


class DealMeta():
    
    def __init__(self, dealer, vuln, level, strain, doubled, redoubled, declarer, tricks_made):
        self.dealer = dealer
        self.vuln = vuln
        self.level = level
        self.strain = strain
        self.doubled = doubled
        self.redoubled = redoubled
        self.declarer = declarer
        self.tricks_made = tricks_made
        
    @classmethod
    def from_str(cls, s):
        #W ALL 3N.-1.N
        parts = s.strip().split()
        outcome = parts[2]
        outcome_parts = outcome.split('.')
        level = int(outcome_parts[0][0])
        doubled = 'X' in outcome_parts[0]
        redoubled = 'XX' in outcome_parts[0]
        strain = outcome_parts[0][1]
        tricks_made = (level + 6) if outcome_parts[1] == '=' else (level + 6) + int(outcome_parts[1])
        declarer = outcome_parts[2]
        
        return cls(
            dealer=parts[0],
            vuln=parts[1],
            level=level,
            strain=strain,
            doubled=doubled,
            redoubled=redoubled,
            declarer=declarer,
            tricks_made=tricks_made
        )
        
    def leader(self):
        return seats[(seat_index[self.declarer] + 1) % 4]
    
    def dealer_relative(self):
        return (seat_index[self.dealer] - seat_index[self.leader()]) % 4
    
    def declarer_vuln(self):
        if self.vuln == 'ALL':
            return True
        if self.vuln == '-':
            return False
        return self.declarer in self.vuln
    
    def leader_vuln(self):
        if self.vuln == 'ALL':
            return True
        if self.vuln == '-':
            return False
        return self.leader() in self.vuln
    
    def get_n_pad_start(self):
        dealer_ix = seat_index[self.dealer]
        declarer_ix = seat_index[self.declarer]
        
        return (dealer_ix - declarer_ix) % 4
    
    def to_dict(self):
        return {
            'dealer': self.dealer,
            'vuln': self.vuln,
            'level': self.level,
            'strain': self.strain,
            'doubled': self.doubled,
            'decarer': self.declarer,
            'tricks_made': self.tricks_made,
            'leader': self.leader(),
            'declarer_vuln': self.declarer_vuln(),
            'leader_vuln': self.leader_vuln()
        }


def binary_hand(suits):
    x = np.zeros((1, 52), np.float16)
    assert(len(suits) == 4)
    for suit_index in [0, 1, 2, 3]:
        for card in suits[suit_index]:
            card_index = card_index_lookup[card]
            x[0, suit_index * 13 + card_index] = 1
    assert(np.sum(x) == 13)
    return x

def loss_array(card_tricks, max_loss=40):
    '''
    ### Prediction loss encoding

    shape = (1, 52)

    for every card there is a loss. cards that you don't have in your hand have a very large loss
    '''
    assert(len(card_tricks) == 13)
    x = max_loss * np.ones((1, 52), np.float16)
    max_tricks = max(card_tricks.values())
    for card, tricks in card_tricks.items():
        loss = max_tricks - tricks
        suit_index = suit_index_lookup[card[0]]
        card_index = card_index_lookup[card[1]]
        x[0, suit_index * 13 + card_index] = loss
    
    assert(np.sum(x < max_loss) == 13)
    
    return x

def binary_auction(auction, n_bids=24):
    '''
    ### Auction sequence encoding
    shape = (n_examples=?, n_bids=24, 40). the last dimension is the bid one-hot encoding
    '''
    assert(len(auction) % 4 == 0)
    auction = auction if len(auction) < n_bids else auction[(len(auction) - n_bids):]
    auction = (['PAD_START'] * (n_bids - len(auction))) + auction
    assert(len(auction) == n_bids)
    
    x = np.zeros((1, n_bids, 40), np.float16)
    
    for i, bid in enumerate(auction):
        x[0, i, :] = bidding.encode_bid(bid)
        
    assert(np.sum(x) == n_bids)
    
    return x

def binary_lead(card_str):
    assert(len(card_str) == 2)

    x = np.zeros((1, 52), np.float16)

    i = suit_index_lookup[card_str[0]] * 13 + card_index_lookup[card_str[1]]
    x[0, i] = 1

    return x


def hand_features(hands, deal_meta):
    '''
    ### hand features encoding

    shape = (1, 1 + 5 + 1 + 1 + 4 + 1 + 1 + 4 + 52) = (1, 70)

    - 0 = encoded level (actual level - 3)
    - 1,2,3,4,5 = strain one-hot (N, S, H, D, C)
    - 6 = doubled
    - 7 = redoubled
    - 8, 9, 10, 11 = dealer id (0=leader, 1=dummy, 2=partner, 3=declarer)
    - 12 = vuln leader
    - 13 = vuln declarer
    - 14, 15, 16, 17 = shape of hand (S, H, D, C)
    - 18:70 = cards themselves one-hot encoded, 52 features with 13 set to 1
    '''
    x = np.zeros((1, 70), np.float16)
    x[0, 0] = deal_meta.level - 3
    if deal_meta.strain == 'N':
        x[0, 1] = 1
    else:
        x[0, 2 + suit_index_lookup[deal_meta.strain]] = 1
    if deal_meta.doubled:
        x[0, 6] = 1
    if deal_meta.redoubled:
        x[0, 7] = 1
    x[0, 8 + deal_meta.dealer_relative()] = 1
    if deal_meta.leader_vuln():
        x[0, 12] = 1
    if deal_meta.declarer_vuln():
        x[0, 13] = 1
    h = binary_hand(hands[seat_index[deal_meta.leader()]])
    shape = (np.sum(h.reshape((4, 13)), axis=1) - 3.25) / 1.75
    x[0, 14:18] = shape
    x[0, 18:70] = h
    
    return x


def lead_data_iterator(fin):
    lines = []
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 17 == 0 and i > 0:
            deal_str = lines[0]
            hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
            deal_meta = DealMeta.from_str(lines[1])
            # auction
            padded_auction = (['PAD_START'] * deal_meta.get_n_pad_start()) + convert_auction(lines[2])
            n_pad_end = 4 - (len(padded_auction) % 4) if (len(padded_auction) % 4) > 0 else 0
            padded_auction = padded_auction + (['PAD_END'] * n_pad_end)
            padded_auction = padded_auction[:-4] if set(padded_auction[-4:]) == set(['PASS', 'PAD_END']) else padded_auction
            # lead_tricks
            lead_tricks = {}
            for card_tricks_line in lines[4:]:
                card, tricks = card_tricks_line.strip().split()
                lead_tricks[card] = int(tricks)
                
            yield lines, hands, deal_meta, padded_auction, lines[3][:2], lead_tricks
            
            lines = []
        
        lines.append(line)


if __name__ == '__main__':
    n = int(sys.argv[1])
    out_path = sys.argv[2]

    n_bids = 24  # maximun 24 bids in the auction

    A = np.zeros((n, n_bids, 40), np.float16)  # auction sequence
    H = np.zeros((n, 70), np.float16)   # hand features
    C = np.zeros((n, 52), np.float16)   # prediction loss
    L = np.zeros((n, 52), np.float16)   # jack's leads

    for i, (lines, hands, deal_meta, padded_auction, jack_lead, lead_tricks) in enumerate(lead_data_iterator(sys.stdin)):
        if i % 1000 == 0:
            sys.stderr.write('%d\n' % i)
        A[i, :, :] = binary_auction(padded_auction)
        H[i, :] = hand_features(hands, deal_meta)
        C[i, :] = loss_array(lead_tricks)
        L[i, :] = binary_lead(jack_lead)

    np.save(os.path.join(out_path, 'A.npy'), A)
    np.save(os.path.join(out_path, 'H.npy'), H)
    np.save(os.path.join(out_path, 'C.npy'), C)
    np.save(os.path.join(out_path, 'L.npy'), L)
