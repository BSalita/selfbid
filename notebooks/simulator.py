
import bidding

from binary_sayc import DealData


class Simulator(object):
    
    def __init__(self, deal_data, contracts, model):
        self.deal_data = deal_data
        self.contracts = contracts
        
        self.model = model
        
        self.par = bidding.get_par(self.contracts, self.deal_data.vuln_ns, self.deal_data.vuln_ew)
        
        self.cache = {}
        
    def simulate_bid(self, auction, n=1, max_bid=False):
        self.deal_data.auction = auction

        i = len(auction) % 4
        padded_auction = (['PAD_START'] * max(0, 3 - len(auction))) + auction
        auction_key = tuple(auction)
        if auction_key not in self.cache:
            
            X, _ = self.deal_data.get_binary(16)

            n_time_steps = len([bid for bid in auction if bid != 'PAD_START']) // 4
            x_input = X[i:i+1, 0:n_time_steps+1]

            out_bid_np = self.model(x_input)

            self.cache[auction_key] = out_bid_np
        else:
            out_bid_np = self.cache[auction_key]
        bids = []
        last_contract = bidding.get_contract(padded_auction)
        while len(bids) < n:
            s_bid = bidding.bid_max_bid(auction, out_bid_np) if max_bid else bidding.sample_bid(padded_auction, out_bid_np)
            bids.append(s_bid)
        return bids

    def simulate_auction(self, auction):
        sim_auction = auction[:]
        while not bidding.auction_over(sim_auction):
            bids = self.simulate_bid(sim_auction, 1)
            sim_auction.append(bids[0])
        return sim_auction
    
    def best_bid(self, auction, n=100):
        results = {}
        declarer2i = {seat:i for i, seat in enumerate(['N', 'E', 'S', 'W'])}
        bids = self.simulate_bid(auction, n)
        vul_i = [int(self.deal_data.vuln_ns), int(self.deal_data.vuln_ew)]
        for bid in bids:
            sim_auction = self.simulate_auction(auction + [bid])
            sim_contract = bidding.get_contract(sim_auction)
            if sim_contract is not None:
                seat_to_bid = len(auction) % 4
                declarer_seat = declarer2i[sim_contract[-1]]
                sign = 1 if (seat_to_bid + declarer_seat) % 2 == 0 else -1
                score = sign * self.contracts.get(sim_contract, (0, 0))[vul_i[declarer_seat % 2]]
            else:
                score = 0
            if bid not in results:
                results[bid] = [0, 0]
            results[bid][0] += score
            results[bid][1] += 1
        max_score_bid = max(((v[0] / v[1], v[1], k) for k, v in results.items()))
        return max_score_bid[0], max_score_bid[2]
    
    def best_auction(self, auction, n=100):
        self.cache = {}
        best_auction = auction[:]
        while not bidding.auction_over(best_auction):
            score, bid = self.best_bid(best_auction, n)
            best_auction.append(bid)
        return score, best_auction
