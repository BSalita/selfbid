import unittest
import numpy as np

from player import follow_suit, get_trick_winner_i
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


class TrickWinnerTest(unittest.TestCase):

    def test_get_trick_winner_i(self):
        trick_cards = np.array([
            [0, 1, 2, 3],
            [15, 15, 8, 9],
            [20, 19, 18, 8],
            [20, 19, 18, 8],
            [20, 19, 18, 8],
            [0, 8, 16, 24],
            [7, 31, 30, 28],
            [15, 7, 24, 6],
            [15, 6, 24, 7],
        ])
        strain = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ])

        trick = np.zeros((trick_cards.shape[0], 4*32))
        s_all = np.array([True] * trick.shape[0])
        for i in range(4):
            trick_i = np.zeros((trick_cards.shape[0], 32))
            trick_i[s_all,trick_cards[:,i]] = 1
            trick[:,i*32:(i+1)*32] = trick_i

        expected_result = np.array([0, 2, 2, 3, 2, 0, 3, 3, 1]).reshape((-1, 1))

        self.assertTrue(np.all(get_trick_winner_i(trick, strain) == expected_result))


if __name__ == '__main__':
    unittest.main()
