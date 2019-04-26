import sys
import gzip
import os.path
import numpy as np

import bidding

from data_access import card_index_lookup


class DealData(object):

    def __init__(self, dealer, vuln_ns, vuln_ew, hands, auction):
        self.dealer = dealer
        self.vuln_ns = vuln_ns
        self.vuln_ew = vuln_ew
        self.hands = list(map(parse_hand, hands))
        self.shapes = list(map(lambda shape: (shape - 3.25)/1.75, map(get_shape, hands)))
        self.hcp = list(map(lambda point_count: (np.array([[point_count]]) - 10) / 4, map(get_hcp, hands)))
        self.auction = auction

    @classmethod
    def from_deal_auction_string(cls, deal_str, auction_str):
        dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
        hands = deal_str.strip().replace('  ', '\t').split('\t')
        auction_parts = auction_str.strip().replace('P', 'PASS').split()
        dealer_ix = dealer[auction_parts[0]]
        vuln_ns, vuln_ew = vuln[auction_parts[1]]
        auction = (['PAD_START'] * dealer_ix) + auction_parts[2:]

        return cls(dealer_ix, vuln_ns, vuln_ew, hands, auction)

    def reset_auction(self):
        self.auction = [bid for bid in self.auction if bid == 'PAD_START']

    def get_binary(self, n_steps=8):
        X = np.zeros((4, n_steps, 2 + 1 + 4 + 52 + 3 * 40), dtype=np.float16)
        y = np.zeros((4, n_steps, 40), dtype=np.float16)

        padded_auction = self.auction + (['PAD_END'] * 4 * n_steps)

        times_seen = [0, 0, 0, 0]

        i = 0
        while sum(times_seen) < 4 * n_steps:
            if padded_auction[i] == 'PAD_START':
                i += 1
                continue

            hand_ix = i % 4

            t = times_seen[hand_ix]
        
            v_we = self.vuln_ns if hand_ix % 2 == 0 else self.vuln_ew
            v_them = self.vuln_ew if hand_ix % 2 == 0 else self.vuln_ns
            vuln = np.array([[v_we, v_them]], dtype=np.float32)
            hcp = self.hcp[hand_ix]
            shape = self.shapes[hand_ix]
            
            lho_bid = padded_auction[i - 3] if i - 3 >= 0 else 'PAD_START'
            partner_bid = padded_auction[i - 2] if i - 2 >= 0 else 'PAD_START'
            rho_bid = padded_auction[i - 1] if i - 1 >= 0 else 'PAD_START'
            target_bid = padded_auction[i]

            ftrs = np.concatenate((
                vuln,
                hcp,
                shape,
                self.hands[hand_ix],
                bidding.encode_bid(lho_bid),
                bidding.encode_bid(partner_bid),
                bidding.encode_bid(rho_bid)
            ), axis=1)

            X[hand_ix, t, :] = ftrs
            y[hand_ix, t, :] = bidding.encode_bid(target_bid)

            times_seen[hand_ix] += 1
            i += 1

        for n in times_seen:
            assert n == n_steps

        return X, y


def parse_hand(hand):
    x = np.zeros((1, 52))
    suits = hand.split()
    assert(len(suits) == 4)
    for suit_index in [0, 1, 2, 3]:
        for card in suits[suit_index][1:]:
            card_index = card_index_lookup[card]
            x[0, suit_index * 13 + card_index] = 1
    return x

def get_shape(hand):
    suits = hand.split()
    return np.array([len(suit) - 1 for suit in suits]).reshape((1, 4))

def get_hcp(hand):
    hcp = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum([hcp.get(c, 0) for c in hand])


def load_deals(fin):
    contracts = {}
    deal_str = ''
    auction_str = ''

    for line_number, line in enumerate(fin):
        line = line.decode('ascii').strip()
        if line_number % 422 == 0:
            if line_number > 0:
                yield (deal_str, auction_str, contracts)
                contracts = {}
                deal_str = ''
                auction_str = ''
            deal_str = line
        elif line_number % 422 == 1:
            auction_str = line
        else:
            cols = line.split('\t')
            contracts[cols[0]] = (int(cols[1]), int(cols[2]))

    yield (deal_str, auction_str, contracts)

def load_deals_no_contracts(fin):
    deal_str = ''
    auction_str = ''

    for line_number, line in enumerate(fin):
        line = line.decode('ascii').strip()
        if line_number % 2 == 0:
            if line_number > 0:
                yield (deal_str, auction_str, None)
                deal_str = ''
                auction_str = ''
            deal_str = line
        elif line_number % 2 == 1:
            auction_str = line

    yield (deal_str, auction_str, None)


def test_debug():
    deal_data = DealData.from_deal_auction_string(
        ' SK76 H965 DQ63 C9854  SAT83 HAJ DJ9754 C73  SQ42 H84 DT2 CAKQT62  SJ95 HKQT732 DAK8 CJ',
        ' N None P P 1C 1H P P 2C 2H P P P'
    )
    print(deal_data.dealer)
    print(deal_data.vuln_ns)
    print(deal_data.vuln_ew)
    print(deal_data.hands)
    print(deal_data.shapes)
    print(deal_data.hcp)
    print(deal_data.auction)

    X, y = deal_data.get_binary()

    import pdb; pdb.set_trace()


def create_binary(data_it, n, out_dir, n_steps=4):
    X = np.zeros((4 * n, n_steps, 2 + 1 + 4 + 52 + 3 * 40), dtype=np.float16)
    y = np.zeros((4 * n, n_steps, 40), dtype=np.float16)

    for i, (deal_str, auction_str, _) in enumerate(data_it):
        if i % 1000 == 0:
            print(i)
        deal_data = DealData.from_deal_auction_string(deal_str, auction_str)
        x_part, y_part = deal_data.get_binary(n_steps)
        start_ix = i * 4
        end_ix = (i + 1) * 4
        X[start_ix:end_ix,:,:] = x_part
        y[start_ix:end_ix,:,:] = y_part

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)


if __name__ == '__main__':
    n = int(sys.argv[1])
    infnm = sys.argv[2]

    #create_binary(load_deals(gzip.open(infnm)), n, '.')
    #create_binary(load_deals_no_contracts(gzip.open(infnm)), n, './bw5c_bin')
    #create_binary(load_deals_no_contracts(gzip.open(infnm)), n, './jos_bin')

    create_binary(load_deals_no_contracts(gzip.open(infnm)), n, './bw5c_8_bin', n_steps=8)

