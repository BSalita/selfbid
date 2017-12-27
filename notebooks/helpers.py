import numpy as np


def encode_bid(bid):
    bid_one_hot = np.zeros((1, n_bids), dtype=np.float32)
    bid_one_hot[0, data.bid2id[bid]] = 1
    return bid_one_hot

def get_input(lho_bid, partner_bid, rho_bid, hand, v_we, v_them):
    vuln = np.array([[v_we, v_them]], dtype=np.float32)
    return np.concatenate((vuln, encode_bid(lho_bid), encode_bid(partner_bid), encode_bid(rho_bid), hand), axis=1)

