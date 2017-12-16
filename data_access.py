import numpy as np

levels = [1, 2, 3, 4, 5, 6, 7]

suits = ['C', 'D', 'H', 'S', 'N']

bid2id = {
    'PAD_START': 1,
    'PAD_END': 2,
    'PASS': 3,
    'X': 4,
    'XX': 5,
}

suitbid2id = {bid:(i+6) for (i, bid) in enumerate(['{}{}'.format(level, suit) for level in levels for suit in suits])}

bid2id.update(suitbid2id)

id2bid = {bid:i for i, bid in bid2id.items()}

card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

def set_data(X, i, deal_str):
    hands = deal_str.split('\t')
    assert(len(hands) == 4)

    for hand_index in [0, 1, 2, 3]:
        assert(len(hands[hand_index]) == 20)
        suits = hands[hand_index].split()
        assert(len(suits) == 4)
        for suit_index in [0, 1, 2, 3]:
            for card in suits[suit_index][1:]:
                card_index = card_index_lookup[card]
                X[i, suit_index, card_index, hand_index] = 1


