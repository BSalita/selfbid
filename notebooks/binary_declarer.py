import sys
import os
import os.path
import numpy as np

from data_access import card_index_lookup
from lead_binary import DealMeta, seats, seat_index, suit_index_lookup


card_index_lookup_x = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7],
    )
)


def binary_hand(suits):
    x = np.zeros((1, 32), np.float16)
    assert(len(suits) == 4)
    for suit_index in [0, 1, 2, 3]:
        for card in suits[suit_index]:
            card_index = card_index_lookup_x[card]
            x[0, suit_index * 8 + card_index] += 1
    assert(np.sum(x) == 13)
    return x


def get_cards(play_str):
    cards = []
    i = 0
    while i < len(play_str):
        cards.append(play_str[i:i+2])
        i += 2
    return cards


def get_tricks(cards):
    return list(map(list, np.array(cards).reshape((13, 4))))


def get_card_index(card):
    suit, value = card[0], card[1]
    return suit_index_lookup[suit] * 8 + card_index_lookup_x[value]


def wins_trick_index(trick, trump, lead_index):
    led_suit = trick[0][0]
    card_values = []
    for card in trick:
        suit, value = card[0], 14 - card_index_lookup[card[1]]
        if suit == trump:
            card_values.append(value + 13)
        elif suit == led_suit:
            card_values.append(value)
        else:
            card_values.append(0)
    return (np.argmax(card_values) + lead_index) % 4


def get_play_labels(play_str, trump):
    tricks = get_tricks(get_cards(play_str))
    
    trick_ix, leads, on_play, cards_in, labels = [], [], [], [], []
    
    lead_index = 0
    card_buffer = ['>>']
    for trick_i, trick in enumerate(tricks):
        win_index = wins_trick_index(trick, trump, lead_index)
        
        for i, card in enumerate(trick):
            player_i = (lead_index + i) % 4
            if player_i % 2 == 1: # declaring side is on play:
                labels.append(card)
                leads.append(lead_index)
                trick_ix.append(trick_i)
                on_play.append(player_i)
                cards_in.append(card_buffer[-2:])
            
            card_buffer.append(card)

        lead_index = win_index
        
    return trick_ix, leads, on_play, cards_in, labels


def binary_data(deal_str, outcome_str, play_str):
    '''
    ## input
    0:32 dummy
    32:64 declarer
    64:96 card
    96:128 card
    128:132 lead one hot 4
    132:136 on play one hot 4
    137 level
    138:143 strain one hot
    144 doubled
    145 declarer vuln
    146 leader vuln

    ## label
    0:32 card
    '''
    x = np.zeros((1, 26, 147), np.float16)
    y = np.zeros((1, 26, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = seat_index[d_meta.declarer]
    dummy_i = (declarer_i + 2) % 4

    dummy_bin = binary_hand(hands[dummy_i])
    declarer_bin = binary_hand(hands[declarer_i])
    
    _, on_leads, on_plays, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain)
    
    for i, (on_lead, on_play, cards_in, card_out) in enumerate(zip(on_leads, on_plays, cards_ins, card_outs)):
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 137] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 138] = 1
        else:
            x[0, i, 139 + suit_index_lookup[d_meta.strain]] = 1
        x[0, i, 144] = int(d_meta.doubled)
        x[0, i, 145] = int(d_meta.declarer_vuln())
        x[0, i, 146] = int(d_meta.leader_vuln())
        x[0, i, 128 + on_lead] = 1
        x[0, i, 132 + on_play] = 1
        x[0, i, 0:32] = dummy_bin[:,:]
        x[0, i, 32:64] = declarer_bin[:,:]
        if cards_in[0] != '>>':
            x[0, i, 64 + get_card_index(cards_in[0])] = 1
        x[0, i, 96 + get_card_index(cards_in[1])] = 1

        # removing cards from declarer and dummy
        if on_play == 1: # dummy on play
            dummy_bin[0, label_card_ix] -= 1
        elif on_play == 3: # declarer on play
            declarer_bin[0, label_card_ix] -= 1
        else:
            raise Exception("either dummy or declarer should be on play")
        
    return x, y


def play_data_iterator(fin):
    lines = []
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 4 == 0 and i > 0:
            yield (lines[0], lines[1], lines[3])
            lines = []

        lines.append(line)

    yield (lines[0], lines[1], lines[3])


if __name__ == '__main__':
    n = int(sys.argv[1])
    out_path = sys.argv[2]

    X = np.zeros((n, 26, 147), np.float16)
    Y = np.zeros((n, 26, 32), np.float16)

    for i, (deal_str, outcome_str, play_str) in enumerate(play_data_iterator(sys.stdin)):
        if i % 1000 == 0:
            print(i)

        x_i, y_i = binary_data(deal_str, outcome_str, play_str)

        X[i, :, :] = x_i
        Y[i, :, :] = y_i

    np.save(os.path.join(out_path, 'X.npy'), X)
    np.save(os.path.join(out_path, 'Y.npy'), Y)
