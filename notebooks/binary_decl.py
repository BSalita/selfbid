import sys
import os
import os.path
import numpy as np

from lead_binary import DealMeta, seats, seat_index, suit_index_lookup

from binary_righty import binary_hand, get_card_index, encode_card, get_play_labels, play_data_iterator

def binary_data(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = seat_index[d_meta.declarer]
    dummy_i = (declarer_i + 2) % 4
    me_i = declarer_i

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 3)
    
    dummy_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_dummy_card = last_trick[1]
        if last_trick_dummy_card not in dummy_played_cards:
            dummy_bin[get_card_index(last_trick_dummy_card)] -= 1
            dummy_played_cards.add(last_trick_dummy_card)
        
        if cards_in[1] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[1])] -= 1
            dummy_played_cards.add(cards_in[1])
        
        x[0, i, 32:64] = dummy_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = encode_card(last_trick[0])
        x[0, i, 96:128] = encode_card(last_trick[1])
        x[0, i, 128:160] = encode_card(last_trick[2])
        x[0, i, 160:192] = encode_card(last_trick[3])
        
        x[0, i, 192:224] = encode_card(cards_in[0])
        x[0, i, 224:256] = encode_card(cards_in[1])
        x[0, i, 256:288] = encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

if __name__ == '__main__':
    n = int(sys.argv[1])
    out_path = sys.argv[2]

    X = np.zeros((n, 11, 298), np.float16)
    Y = np.zeros((n, 11, 32), np.float16)

    for i, (deal_str, outcome_str, play_str) in enumerate(play_data_iterator(sys.stdin)):
        if i % 1000 == 0:
            print(i)

        x_i, y_i = binary_data(deal_str, outcome_str, play_str)

        X[i, :, :] = x_i
        Y[i, :, :] = y_i

    np.save(os.path.join(out_path, 'X.npy'), X)
    np.save(os.path.join(out_path, 'Y.npy'), Y)
