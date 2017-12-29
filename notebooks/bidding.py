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
        bid_one_hot = np.random.multinomial(1, from_bids)
        bid_id = np.argmax(bid_one_hot)
        bid = ID2BID[bid_id]
        if can_bid(bid, auction):
            return bid

def bid_max_bid(auction, from_bids):
    bid = ID2BID(np.argmax(from_bids))
    if can_bid(bid, auction):
        return bid
    else:
        print('invalid bid', auction, bid)
        return sample_bid(auction, from_bids)
        
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
