import unittest
import numpy as np

from player import follow_suit
from binary import get_binary_hand_from_cards, get_cards_from_binary_hand


class FollowSuitTest(unittest.TestCase):
    
    def test_follow_suit(self):
        cards_softmax = np.random.random((8, 32))
        own_cards = np.array(list(map(get_binary_hand_from_cards, [
            [0, 1, 2, 7, 7, 7, 8, 11, 12, 16, 24, 31],
            [0, 1, 2, 7, 7, 7, 8, 11, 12, 16, 24, 31],  
            [24, 25, 26, 27, 28, 29, 30, 31, 31],  # we only have clubs
            [0, 1, 8, 9, 16, 17, 24, 25],
            [0, 1, 16, 17, 24, 25],     # we don't have hearts
            [],   # we don't have any cards
            [8, 9, 10, 16, 17, 18, 24, 31, 31, 31],   # we don't have any spades
            [],   # we don't have any cards
        ])))
        trick_suit = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        legal_cards_softmax = follow_suit(cards_softmax, own_cards, trick_suit)

        self.assertTrue(
            (np.abs(np.sum(legal_cards_softmax, axis=1) - np.array([1, 1, 1, 1, 1, 0, 1, 0])) < 1e-9).all())

        self.assertTrue(
            ((legal_cards_softmax[0,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([0, 1, 2, 7, 8, 11, 12, 16, 24, 31])).all())

        self.assertTrue(
            ((legal_cards_softmax[1,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([8, 11, 12])).all())

        self.assertTrue(
            ((legal_cards_softmax[2,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([24, 25, 26, 27, 28, 29, 30, 31])).all())

        self.assertTrue(
            ((legal_cards_softmax[3,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([16, 17])).all())

        self.assertTrue(
            ((legal_cards_softmax[4,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([0, 1, 16, 17, 24, 25])).all())

        self.assertTrue(
            ((legal_cards_softmax[6,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([8, 9, 10, 16, 17, 18, 24, 31])).all())

        for i in (5, 7):
            self.assertTrue(
                ((legal_cards_softmax[i,:] > 1e-9).astype(np.int) == get_binary_hand_from_cards([])).all())



if __name__ == '__main__':
    unittest.main()