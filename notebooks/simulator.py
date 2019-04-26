
import numpy as np

import bidding

from binary_sayc import DealData


class Simulator(object):
    
    def __init__(self, deal_data, contracts, model):
        self.deal_data = deal_data
        self.contracts = contracts
        
        self.model = model
        
        self.par = bidding.get_par(self.contracts, self.deal_data.vuln_ns, self.deal_data.vuln_ew)
        
        self.cache = {}
        
    def simulate_bid(self, auction, state_in, n=1, max_bid=False):
        self.deal_data.auction = auction

        i = len(auction) % 4
        padded_auction = (['PAD_START'] * max(0, 3 - len(auction))) + auction
        auction_key = tuple(auction)
        if auction_key not in self.cache:
            
            X, _ = self.deal_data.get_binary(16)

            n_time_steps = len([bid for bid in auction if bid != 'PAD_START']) // 4
            x_input = X[i:i+1, n_time_steps]

            out_bid_np, next_state = self.model(x_input, state_in)

            self.cache[auction_key] = out_bid_np, next_state
        else:
            out_bid_np, next_state = self.cache[auction_key]
        bids = []
        while len(bids) < n:
            s_bid = bidding.bid_max_bid(auction, out_bid_np[0]) if max_bid else bidding.sample_bid(padded_auction, out_bid_np[0])
            bids.append(s_bid)
        return bids, next_state

    def next_bid_np(self, auction, state_in):
        #import pdb; pdb.set_trace()
        self.deal_data.auction = auction

        i = len(auction) % 4

        X, _ = self.deal_data.get_binary(16)

        n_time_steps = len([bid for bid in auction if bid != 'PAD_START']) // 4
        x_input = X[i:i+1, n_time_steps]

        out_bid_np, next_state = self.model(x_input, state_in)

        return out_bid_np, next_state


    def simulate_auction(self, auction, states, max_bid=False):
        #import pdb; pdb.set_trace()

        sim_auction = auction[:]
        sim_states = states[:]
        
        while not bidding.auction_over(sim_auction):
            i = len(sim_auction) % 4
            bids, next_state = self.simulate_bid(sim_auction, sim_states[i], 1, max_bid)
            sim_states[i] = next_state
            sim_auction.append(bids[0])
        return sim_auction

    def simulate_auction_np(self, auction, states, sim_bid_index=-1):
        sim_auction = auction[:]
        auction_out_bids = []
        sim_states = states[:]

        padded_auction = (['PAD_START'] * max(0, 3 - len(auction))) + auction

        bid_index = 0
        while not bidding.auction_over(sim_auction):
            i = len(sim_auction) % 4

            out_bid_np, next_state = self.next_bid_np(sim_auction, sim_states[i])

            sim_states[i] = next_state

            auction_out_bids.append(out_bid_np)

            if bid_index == sim_bid_index:
                # set the max bid to 0 and sample
                out_bid_np[0, np.argmax(out_bid_np)] = 0
                sim_auction.append(bidding.sample_bid(padded_auction, out_bid_np[0]))
            else:
                sim_auction.append(bidding.bid_max_bid(auction, out_bid_np[0]))

            bid_index += 1

        return sim_auction, auction_out_bids

    
    def best_bid(self, auction, states, n=100):
        sim_states = states[:]
        results = {}
        declarer2i = {seat:i for i, seat in enumerate(['N', 'E', 'S', 'W'])}
       
        hand_i = len(auction) % 4
        bids, next_state = self.simulate_bid(auction, sim_states[hand_i], n)
        sim_states[hand_i] = next_state

        vul_i = [int(self.deal_data.vuln_ns), int(self.deal_data.vuln_ew)]
        for bid in bids:
            sim_auction = self.simulate_auction(auction + [bid], sim_states, True)
            sim_contract = bidding.get_contract(sim_auction)
            if sim_contract is not None:
                seat_to_bid = len(auction) % 4
                declarer_seat = declarer2i[sim_contract[-1]]
                sign = 1 if (seat_to_bid + declarer_seat) % 2 == 0 else -1
                score = sign * self.contracts.get(sim_contract, (0, 0))[vul_i[declarer_seat % 2]]
            else:
                score = 0
            if bid not in results:
                results[bid] = []
            results[bid].append(score)
        max_score_bid = max(( (min(v), k) for k, v in results.items()))
        return max_score_bid[0], max_score_bid[1]
    
    def best_auction(self, auction, states, n=100):
        self.cache = {}
        best_auction = auction[:]
        while not bidding.auction_over(best_auction):
            score, bid = self.best_bid(best_auction, states, n)
            best_auction.append(bid)
        return score, best_auction


    def get_options(self, auction, bid_np):
        result = []
        done = False
        attempt = 0
        seen_bids = set()
        while not done:
            if len(result) > 3:
                break
            if attempt > 20:
                #print("no bid found", auction)
                break
            attempt += 1
            i = np.argmax(bid_np)
            p = bid_np[i]
            bid_np[i] = 0
            bid = bidding.ID2BID[i]
            
            if bidding.can_bid(bid, auction):
                if len(result) == 0:
                    if bid not in seen_bids:
                        result.append((bid, p))
                        seen_bids.add(bid)
                elif len(result) == 1:
                    if p > 0.1 and bid not in seen_bids:
                        result.append((bid, p))
                        seen_bids.add(bid)
                    else:
                        done = True
                else:
                    if p > 0.2 and bid not in seen_bids:
                        result.append((bid, p))
                        seen_bids.add(bid)
                    else:
                        done = True

        # also consider X
        contract = bidding.get_contract(auction)
        # if contract is not None and contract.startswith('4N'):
        #     import pdb; pdb.set_trace()
            
        if contract is not None and bidding.is_higher_contract(contract, '3S') and self.contracts[contract][1] < -100 and bidding.can_double(auction):
            if 'X' not in seen_bids:
                result.append(('X', 0))
                seen_bids.add('X')

        if 'PASS' not in seen_bids:
            result.append(('PASS', 0))
            seen_bids.add('PASS')

        assert len(result) > 0
        assert len(result) == len(seen_bids)

        return result

    def best(self, auction, states, best_score_ns, best_score_ew, potential_ns, potential_ew):
        if bidding.auction_over(auction):
            score = bidding.get_score(bidding.get_contract(auction), self.contracts, self.deal_data.vuln_ns, self.deal_data.vuln_ew)
            #print('returning', auction, score)
            return auction, score
        
        states_copy = states[:]
        
        hand_i = len(auction) % 4
        side = hand_i % 2
        is_side_vuln = [self.deal_data.vuln_ns, self.deal_data.vuln_ew][side]
        agg_fun = [max, min][side]

        self.deal_data.auction = auction
        X, _ = self.deal_data.get_binary(16)

        n_time_steps = len([bid for bid in auction if bid != 'PAD_START']) // 4
        x_input = X[hand_i:hand_i+1, n_time_steps]

        bid_np, next_state = self.model(x_input, states_copy[hand_i])

        states_copy[hand_i] = next_state

        b_auc, b_score_ns, b_score_ew, b_p = None, best_score_ns, best_score_ew, 0
        
        # if tuple(auction) == ('1H', 'PASS', '2N', 'PASS', '3S', 'PASS'):
        #     import pdb; pdb.set_trace()

        options = self.get_options(auction, bid_np[0])
        #print("options", options)
        
        for bid, p in options:

            contract_so_far = bidding.get_contract(auction + [bid])

            new_potential_ns = [(c, s) for c, s in potential_ns if not bidding.is_higher_contract(contract_so_far, c)] if contract_so_far is not None else potential_ns
            new_potential_ew = [(c, s) for c, s in potential_ew if not bidding.is_higher_contract(contract_so_far, c)] if contract_so_far is not None else potential_ew
            
            # [(contract_so_far, self.contracts[contract_so_far][int(is_side_vuln)])] + 
            best_possible = best_possible_score((new_potential_ns if side == 0 else new_potential_ew), agg_fun)
            best_so_far = b_score_ns if side == 0 else b_score_ew

            #print(auction, bid)
            
            if not is_better(best_possible, best_so_far, side):
                #print('{} found nothing better than {}, best possible {}'.format(side, best_so_far, best_possible))
                if bid not in ('PASS', 'X'):
                    continue


            #print('{} best so far {}, best possible {}'.format(side, best_so_far, best_possible))
            
            bid_auction, bid_score = self.best(auction + [bid], states_copy, b_score_ns, b_score_ew, new_potential_ns, new_potential_ew)        

            if b_auc is None:
                b_auc, b_score_ns, b_score_ew, b_p = bid_auction, bid_score, bid_score, p
                continue

            same_score = (bid_score == best_so_far)
            if same_score:
                if p > b_p:
                    b_auc, b_score_ns, b_score_ew, b_p = bid_auction, bid_score, bid_score, p
                    continue

            #better_score = (bid_score > b_score) if hand_i % 2 == 0 else (bid_score < b_score)
            if is_better(bid_score, best_so_far, side):
                b_auc, b_score_ns, b_score_ew, b_p = bid_auction, bid_score, bid_score, p

        #import pdb; pdb.set_trace()

        return b_auc, b_score_ns


def get_potential_contracts(contracts, side, vuln):
    declarer_side = {'N': 0, 'E': 1, 'S': 0, 'W': 1}
    sign = 1 if side == 0 else -1
    potential_contracts = {}
    for contract, (score_white, score_red) in contracts.items():
        declarer = contract[-1]
        if side != declarer_side[declarer]:
            continue
        if 'XX' in contract:
            continue
        if 'X' in contract and score_red > 0:
            continue
        if score_red < -200 and 'X' not in contract:
            continue
        side_contract = contract[:-1]
        score = score_red if vuln else score_white
        potential_contracts[side_contract] = max(score, potential_contracts.get(side_contract, -10000))
        
    sorted_potential = sorted(potential_contracts.items(), key=lambda cs: (int(cs[0][0]), bidding.SUIT_RANK[cs[0][1]], cs[0][2:]))
    sign_potential = [(c, sign*s) for c, s in sorted_potential]
    return sign_potential


def best_possible_score(potential_contract_scores, agg_fun):
    assert agg_fun in (min, max)
    return agg_fun(potential_contract_scores, key=lambda cs: cs[1])[1]

def is_better(new_score, old_score, side):
    if side == 0:
        return new_score > old_score
    else:
        return new_score < old_score
