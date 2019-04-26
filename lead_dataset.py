import sys

from redeal import *
from redeal.global_defs import Seat

def deal_from_jack(deal_str):
    deal_str = deal_str.strip()[2:]
    [west, north, east, south] = deal_str.split()

    return Deal({
        Seat.W: hand_from_jack(west).cards,
        Seat.N: hand_from_jack(north).cards,
        Seat.E: hand_from_jack(east).cards,
        Seat.S: hand_from_jack(south).cards,
        '_remaining': []
    })
    
def hand_from_jack(hand_str):
    suits = hand_str.split('.')
    s = ' '.join([(s if s else '-') for s in suits])
    return H(s)

declarer_to_leader = {
    'S': 'W',
    'W': 'N',
    'N': 'E',
    'E': 'S'
}

def get_strain_leader(outcome_str):
    parts = outcome_str.split()
    result_parts = parts[2].split('.')
    return (result_parts[0][1], Seat[declarer_to_leader[result_parts[2]]])

def get_all_card_tricks(cards, card_tricks):
    sorted_cards = sorted(cards, reverse=True)
    prev = None
    for c in sorted_cards:
        if c in card_tricks:
            yield (c, card_tricks[c])
            prev = card_tricks[c]
        else:
            yield (c, prev)

def main():
    deal = None
    strain = None
    leader = None

    for i, line in enumerate(sys.stdin):
        if i % 1000 == 0:
            sys.stderr.write('%d\n' % i)
        line = line.strip()
        print(line)
        if i % 4 == 0:
            deal = deal_from_jack(line)
        elif i % 4 == 1:
            strain, leader = get_strain_leader(line)
        elif i % 4 == 2:
            pass  # auction, do nothing
        elif i % 4 == 3:
            # play. this is the last line
            card_tricks = deal.dd_all_tricks(strain, leader.name)
            for c, tricks in get_all_card_tricks(deal[leader].cards(), card_tricks):
                print('%s %d' % (str(c), tricks))

if __name__ == '__main__':
    main()

