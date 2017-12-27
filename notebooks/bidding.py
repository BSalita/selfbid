import numpy as np


LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUITS = ['C', 'D', 'H', 'S', 'N']
SUIT_RANK = {suit:i for i, suit in enumerate(SUITS)}

BID2ID = {
    'PAD_START': 0,
    'PAD_END': 1,
    'PASS': 2,
    'X': 3,
    'XX': 4,
}

SUITBID2ID = {bid:(i+5) for (i, bid) in enumerate(['{}{}'.format(level, suit) for level in LEVELS for suit in SUITS])}

BID2ID.update(SUITBID2ID)

ID2BID = {bid:i for i, bid in BID2ID.items()}


def encode_bid(bid):
    bid_one_hot = np.zeros((1, len(BID2ID)), dtype=np.float32)
    bid_one_hot[0, BID2ID[bid]] = 1
    return bid_one_hot

def get_input(lho_bid, partner_bid, rho_bid, hand, v_we, v_them):
    vuln = np.array([[v_we, v_them]], dtype=np.float32)
    return np.concatenate((vuln, encode_bid(lho_bid), encode_bid(partner_bid), encode_bid(rho_bid), hand), axis=1)

def is_contract(bid):
    return bid[0].isdigit()

def can_double(auction):
    if len(auction) == 0:
        return False
    if is_contract(auction[-1]):
        return True
    if len(auction) >= 3 and is_contract(auction[-3]) and auction[-2] == 'PASS' and auction[-1] == 'PASS':
        return True
    return False

def can_redouble(auction):
    if len(auction) == 0:
        return False
    if auction[-1] == 'X':
        return True
    if len(auction) >= 3 and auction[-3] == 'X' and auction[-2] == 'PASS' and auction[-1] == 'PASS':
        return True
    return False

def last_contract(auction):
    for bid in reversed(auction):
        if is_contract(bid):
            return bid
    return None

def contract_level_step(contract):
    return int(contract[0])*5 + SUIT_RANK[contract[1]]

def is_higher_contract(this_contract, other_contract):
    return contract_level_step(this_contract) > contract_level_step(other_contract)

def can_bid_contract(bid, auction):
    assert is_contract(bid)
    contract = last_contract(auction)
    if contract is None:
        return True
    return is_higher_contract(bid, contract)

def auction_over(auction):
    if len(auction) < 4:
        return False
    if auction[-1] == 'PAD_END':
        return True
    contract = last_contract(auction)
    if contract is None:
        return all([bid == 'PASS' for bid in auction[-4:]]) and all([bid == 'PAD_START' for bid in auction[:-4]])
    else:
        return all([bid == 'PASS' for bid in auction[-3:]])

def can_bid(bid, auction):
    if bid == 'PASS':
        return True
    if bid == 'X':
        return can_double(auction)
    if bid == 'XX':
        return can_redouble(auction)
    if is_contract(bid):
        return can_bid_contract(bid, auction)
    return False

def sample_bid(auction, from_bids):
    from_bids = from_bids / (np.sum(from_bids) + 1e-6)
    if auction_over(auction):
        return 'PAD_END'
    while True:
        bid_one_hot = np.random.multinomial(1, from_bids[0])
        bid_id = np.argmax(bid_one_hot)
        bid = ID2BID[bid_id]
        if can_bid(bid, auction):
            return bid
        
def get_contract(auction):
    contract = None
    doubled = False
    redoubled = False
    last_bid_i = None
    for i in reversed(range(len(auction))):
        bid = auction[i]
        if is_contract(bid):
            contract = bid
            last_bid_i = i
            break
        if bid == 'X':
            doubled = True
        if bid == 'XX':
            redoubled = True
    
    if contract is None:
        return None
    
    declarer_i = None
    for i in range(last_bid_i + 1):
        bid = auction[i]
        if not is_contract(bid):
            continue
        if (i + last_bid_i) % 2 != 0:
            continue
        if bid[1] != contract[1]:
            continue
        declarer_i = i
        break
        
    declarer = ['N', 'E', 'S', 'W'][declarer_i % 4]
    
    xx = '' if not doubled else 'X' if not redoubled else 'XX'
    
    return contract + xx + declarer

def get_par(contracts, vuln_ns, vuln_ew):
    side_vuln = [int(vuln_ns), int(vuln_ew)]
    side = {'N': 0, 'E': 1, 'S': 0, 'W': 1}
    
    contract_scores = sorted(contracts.items(), key=lambda cs: (int(cs[0][0]) * 5 + SUIT_RANK[cs[0][1]], cs[0]))
    
    best_score = [0, 0]
    best_contract = [None, None]
    
    for contract, scores in contract_scores:
        side_i = side[contract[-1]]
        score = scores[side_vuln[side_i]]
        
        if score > best_score[side_i]:
            if score > 0 and 'X' in contract:
                continue
            if score < 0 and 'X' not in contract:
                continue
            best_score[side_i] = score
            best_score[(side_i + 1) % 2] = -score
            best_contract[side_i] = contract
            best_contract[(side_i + 1) % 2] = contract
            
    assert best_contract[0] == best_contract[1]
            
    return best_contract[0]


class Simulator(object):
    
    def __init__(self, deal, contracts, model):
        # TODO: fix vulnerability
        self.deal = deal
        self.contracts = contracts
        self.hands = [
            deal[0,:,:,0].reshape((1, 52)), 
            deal[0,:,:,1].reshape((1, 52)),
            deal[0,:,:,2].reshape((1, 52)),
            deal[0,:,:,3].reshape((1, 52)),
        ]
        self.model = model
        
        self.par = get_par(self.contracts, False, False)
        
        self.cache = {}
        
    def simulate_bid(self, auction, s_c, s_h, n=1):
        i = len(auction) % 4
        padded_auction = (['PAD_START'] * max(0, 3 - len(auction))) + auction
        auction_key = tuple(auction)
        if auction_key not in self.cache:
            # TODO: fix vulnerability
            x_input = get_input(padded_auction[-3], padded_auction[-2], padded_auction[-1], self.hands[i], False, False)
            out_bid_np, next_c_np, next_h_np = self.model(x_input, s_c, s_h)
            self.cache[auction_key] = (out_bid_np, next_c_np, next_h_np)
        else:
            out_bid_np, next_c_np, next_h_np = self.cache[auction_key]
        bids = []
        last_contract = get_contract(padded_auction)
        while len(bids) < n:
            s_bid = sample_bid(padded_auction, out_bid_np)
            if is_contract(s_bid) and is_higher_contract(s_bid, self.par):
                 continue
            if 'X' in s_bid and contract_level_step(last_contract) == contract_level_step(self.par) and last_contract[-1] == self.par[-1]:
                if 'X' not in self.par:
                    continue
            bids.append(s_bid)
        return bids, (next_c_np, next_h_np)

    def simulate_auction(self, auction, s_c, s_h):
        sim_auction = auction[:]
        C, H = s_c, s_h
        while not auction_over(sim_auction):
            bids, (next_c_np, next_h_np) = self.simulate_bid(sim_auction, C, H, 1)
            sim_auction.append(bids[0])
            C = next_c_np
            H = next_h_np
        return sim_auction
    
    def best_bid(self, auction, s_c, s_h, n=100):
        results = {}
        declarer2i = {seat:i for i, seat in enumerate(['N', 'E', 'S', 'W'])}
        bids, (next_c_np, next_h_np) = self.simulate_bid(auction, s_c, s_h, n)
        for bid in bids:
            sim_auction = self.simulate_auction(auction + [bid], next_c_np, next_h_np)
            sim_contract = get_contract(sim_auction)
            if sim_contract is not None:
                seat_to_bid = len(auction) % 4
                declarer_seat = declarer2i[sim_contract[-1]]
                sign = 1 if (seat_to_bid + declarer_seat) % 2 == 0 else -1
                score = sign * self.contracts.get(sim_contract, (0, 0))[0]  # TODO: fix vulnerability
            else:
                score = 0
            if bid not in results:
                results[bid] = [0, 0]
            results[bid][0] += score
            results[bid][1] += 1
        max_score_bid = max(((v[0] / v[1], k) for k, v in results.items()))
        return max_score_bid
    
    def best_auction(self, auction, s_c, s_h, n=100):
        self.cache = {}
        best_auction = auction[:]
        while not auction_over(best_auction):
            score, bid = self.best_bid(best_auction, s_c, s_h, n)
            best_auction.append(bid)
        return score, best_auction
    