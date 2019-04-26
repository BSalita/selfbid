import sys
import gzip

import bidding

from lstm_bidder import Bidder
from binary_sayc import load_deals, DealData


call_trans = {
    'PASS': 'p ',
    'X': 'X ',
    'PAD_START': '  '
}

def disp_auction(auction):
    auction = ['  '] + [call_trans[call] if call in call_trans else call for call in auction]
    print(' '.join(['West', 'North', 'East ', 'South']))
    acc = []
    for i, call in enumerate(auction):
        
        if i % 4 == 0 and i > 0:
            print('    '.join(acc))
            acc = []
        acc.append(call)
    print('    '.join(acc))


def disp_deal(deal_str, dealer, vuln):
    hands = deal_str.split('  ')
    north = [s[1:] for s in hands[0].split(' ')]
    east = [s[1:] for s in hands[1].split(' ')]
    south = [s[1:] for s in hands[2].split(' ')]
    west = [s[1:] for s in hands[3].split(' ')]
    annot = dealer + ' ' + vuln
    print(annot + (8 - len(annot)) * ' ' + north[0])
    print(8 * ' ' + north[1])
    print(8 * ' ' + north[2])
    print(8 * ' ' + north[3])
    print(west[0] + (16 - len(west[0])) * ' ' + east[0])
    print(west[1] + (16 - len(west[1])) * ' ' + east[1])
    print(west[2] + (16 - len(west[2])) * ' ' + east[2])
    print(west[3] + (16 - len(west[3])) * ' ' + east[3])
    print(8 * ' ' + south[0])
    print(8 * ' ' + south[1])
    print(8 * ' ' + south[2])
    print(8 * ' ' + south[3])


if __name__ == '__main__':
    bw5c_bidder = Bidder('bw5c', './bw5c_model/bw5c-500000')
    jos_bidder = Bidder('bw5c_8', './bw5c_8_model/bw5c_8-500000')

    reader = load_deals(gzip.open('../deals_bidding_valid_0001.gz'))

    for i, (deal_str, auction_str, contracts) in enumerate(reader):
        if i > 400:
            break

        auction_parts = auction_str.split()

        deal_data = DealData.from_deal_auction_string(deal_str, auction_str)

        disp_deal(deal_str, auction_parts[0], auction_parts[1])
        print('')

        bw5c_auction = bw5c_bidder.simulate_auction(deal_data)
        #bw5c_contract = bidding.get_contract(bw5c_auction)
        jos_auction = jos_bidder.simulate_auction(deal_data)
        #jos_contract = bidding.get_contract(jos_auction)

        #print('BW5C contract: %s %s' % (bw5c_contract, bidding.get_score(bw5c_contract, contracts, deal_data.vuln_ns, deal_data.vuln_ew)))
        print('BW5C auction:')
        disp_auction(bw5c_auction)
        print('')
        #print('JOS contract: %s %s' % (jos_contract, bidding.get_score(jos_contract, contracts, deal_data.vuln_ns, deal_data.vuln_ew)))
        print('bw5c_8 auction:')
        disp_auction(jos_auction)
        print('\n-------------------------\n')
