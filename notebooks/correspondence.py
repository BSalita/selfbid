import sys
import bidding

import jack_to_deal_auction as jack_format

from lstm_bidder import Bidder
from binary_sayc import DealData


def to_auction_list(auction_str):
    i = 0
    result = []
    while i < len(auction_str):
        result.append(auction_str[i:i+2])
        i += 2
    return result

def parse_line(line):
    hand_metadata = line.strip().split('#:')
    deal_str = jack_format.convert_deal_str(hand_metadata[0])
    metadata_parts = hand_metadata[1].split()
    table = int(metadata_parts[2][2:])
    dealer = metadata_parts[3][2:]
    vuln = metadata_parts[4][2:]
    jack_auction_str = ' '.join(to_auction_list(metadata_parts[5][2:]))
    auction_str = '%s %s %s' % (dealer, jack_format.jack_vuln_lookup[vuln], jack_format.convert_auction_str(jack_auction_str))
    
    return table, DealData.from_deal_auction_string(deal_str, auction_str)


def bid_to_jack_format(bid_str):
    lookup = {
        'PASS': 'PP',
        'X': 'DD',
        'XX': 'RR'
    }
    if bid_str in lookup:
        return lookup[bid_str]
    else:
        return bid_str

def ns_turn(auction):
    return (len(auction) % 4) in (0, 2)

def is_my_turn(team, table, auction):
    if team == 'home':
        if table == 1:
            return ns_turn(auction)
        else:
            return not ns_turn(auction)
    else:
        if table == 1:
            return not ns_turn(auction)
        else:
            return ns_turn(auction)


def correspondence(team):
    bidder = Bidder('bw5c', './bw5c_model/bw5c-500000')

    for line in sys.stdin:
    #for line in open('/home/ldali/datasets/jack/match/Match000.1'):
        line = line.strip()
        table, deal_data = parse_line(line)

        # if '#:1736' in line and 'T:1' in line:
        #     import pdb; pdb.set_trace()

        if is_my_turn(team, table, deal_data.auction) and not bidding.auction_over(deal_data.auction):
            bid = bidder.next_bid(deal_data, deal_data.auction)
            sys.stdout.write(line)
            # if not line.endswith('A:'):
            #     sys.stdout.write(' ')
            sys.stdout.write(bid_to_jack_format(bid))
            sys.stdout.write('\n')
        else:
            print(line)


if __name__ == '__main__':
    team = sys.argv[1]
    correspondence(team)
