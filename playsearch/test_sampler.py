import unittest
import numpy as np

from sampler import distribute_cards


class SamplerTest(unittest.TestCase):

    def test_distribute_cards(self):
        cards = np.array([
            [0, 1, 2, 7, 7, 7, 13, 31, 31, 31],
            [1, 2, 3, 4, 30, 31, 31, 31, 31, 31]
        ])

        expected_hand = np.zeros((2, 32))
        expected_hand[0, 0] = 1
        expected_hand[0, 1] = 1
        expected_hand[0, 2] = 1
        expected_hand[0, 7] = 3
        expected_hand[0, 13] = 1
        expected_hand[0, 31] = 3
        expected_hand[1, 1] = 1
        expected_hand[1, 2] = 1
        expected_hand[1, 3] = 1
        expected_hand[1, 4] = 1
        expected_hand[1, 30] = 1
        expected_hand[1, 31] = 5

        distributed_hand = distribute_cards(cards)

        self.assertTrue((distributed_hand == expected_hand).all())

