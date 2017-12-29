import sys
import gzip
import math

import numpy as np

import bidding

from binary_sayc import load_deals, DealData


def get_score(contract, contract_scores, vuln_ns, vuln_ew):
    if contract is None:
        return 0
    declarer = contract[-1]
    sign = 1 if declarer in ('N', 'S') else -1
    declarer_vuln = int(vuln_ns if declarer in ('N', 'S') else vuln_ew)
    return sign * contract_scores[contract][declarer_vuln]


def eval_bidding(fin):
    errors = []

    reader = load_deals(fin)
    for i, (deal_str, auction_str, contracts) in enumerate(reader):
        if i % 1000 == 0:
            sys.stdout.write('.')
        deal_data = DealData.from_deal_auction_string(deal_str, auction_str)
        par_contract = bidding.get_par(contracts, deal_data.vuln_ns, deal_data.vuln_ew)
        reached_contract = bidding.get_contract(deal_data.auction)

        par_score = get_score(par_contract, contracts, deal_data.vuln_ns, deal_data.vuln_ew)
        reached_score = get_score(reached_contract, contracts, deal_data.vuln_ns, deal_data.vuln_ew)

        errors.append(math.fabs(par_score - reached_score))

    return np.mean(errors)
