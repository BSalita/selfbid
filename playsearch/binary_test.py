import unittest
import numpy as np

from binary import BinaryInput, get_cards_from_binary_hand, get_binary_hand_from_cards


class BinaryInputTest(unittest.TestCase):

    def test_shape(self):
        binary_in = BinaryInput(np.zeros((100, 298)))

        self.assertTrue(binary_in.x.shape == (100, binary_in.n_ftrs))

    def test_hand_get_set(self):
        binary_in = BinaryInput(np.zeros((100, 298)))

        self.assertTrue((binary_in.get_player_hand() == np.zeros((100, 32))).all())
        self.assertTrue((binary_in.get_public_hand() == np.zeros((100, 32))).all())

        player_hand = np.zeros((100, 32)) + 1
        public_hand = np.zeros((100, 32)) + 2

        binary_in.set_player_hand(player_hand)
        self.assertTrue((binary_in.get_player_hand() == player_hand).all())

        binary_in.set_public_hand(public_hand)
        self.assertTrue((binary_in.get_public_hand() == public_hand).all())

    def test_tricks_get_set(self):
        binary_in = BinaryInput(np.zeros((100, 298)))

        self.assertTrue((binary_in.get_last_trick() == np.zeros((100, 4, 32))).all())
        self.assertTrue((binary_in.get_this_trick() == np.zeros((100, 3, 32))).all())

        last_trick = np.zeros((100, 4, 32))
        last_trick[:, 0, :] = 1
        last_trick[:, 1, :] = 2
        last_trick[:, 2, :] = 3
        last_trick[:, 3, :] = 4

        this_trick = np.zeros((100, 3, 32))
        this_trick[:, 0, :] = 7
        this_trick[:, 1, :] = 8
        this_trick[:, 2, :] = 9

        binary_in.set_last_trick(last_trick)
        self.assertTrue((binary_in.get_last_trick() == last_trick).all())

        binary_in.set_this_trick(this_trick)
        self.assertTrue((binary_in.get_this_trick() == this_trick).all())

        with self.assertRaises(ValueError):
            binary_in.set_last_trick(this_trick)

        with self.assertRaises(ValueError):
            binary_in.set_this_trick(last_trick)

    def test_lead_index(self):
        binary_in = BinaryInput(np.zeros((100, 298)))

        lead_i = np.arange(100) % 4

        binary_in.set_last_trick_lead(lead_i)
        self.assertTrue((binary_in.x[0, 288:292] == np.array([1, 0, 0, 0])).all())
        self.assertTrue((binary_in.x[1, 288:292] == np.array([0, 1, 0, 0])).all())
        self.assertTrue((binary_in.x[2, 288:292] == np.array([0, 0, 1, 0])).all())
        self.assertTrue((binary_in.x[3, 288:292] == np.array([0, 0, 0, 1])).all())
        self.assertTrue((binary_in.x[4, 288:292] == np.array([1, 0, 0, 0])).all())

        self.assertTrue((binary_in.get_last_trick_lead() == lead_i).all())

    def test_broadcast_set_player_hand(self):
        binary_in = BinaryInput(np.zeros((100, 298)))

        hand = np.zeros((1, 32))
        hand[0, [1,2,3,4,5,6,7,8,9,10,11,12,13]] = 1

        binary_in.set_player_hand(hand)

        self.assertTrue((binary_in.get_player_hand()[0,:] == hand).all())
        self.assertTrue((binary_in.get_player_hand()[1,:] == hand).all())
        self.assertTrue((binary_in.get_player_hand()[2,:] == hand).all())
        self.assertTrue((binary_in.get_player_hand()[99,:] == hand).all())

    def test_last_trick_lead(self):
        pass

    def test_cards_binary_hand(self):
        hands = [
            np.array([
                [1, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 3],
                [0, 0, 1, 0, 0, 0, 1, 0],
            ]).reshape(32),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 3],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
            ]).reshape(32),
            np.array([
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]).reshape(32),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]).reshape(32),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]).reshape(32),
        ]

        cards_list = [
            np.array([0, 1, 4, 7, 15, 15, 23, 23, 23, 26, 30]),
            np.array([7, 7, 7, 8, 11, 15, 16, 18, 19, 21, 22, 28, 31]),
            np.array([1.0]),
            np.array([7, 7]),
            np.array([]),
        ]

        for hand, cards in zip(hands, cards_list):
            self.assertTrue((get_cards_from_binary_hand(hand) == cards).all())
            self.assertTrue((get_binary_hand_from_cards(cards) == hand).all())


if __name__ == '__main__':
    unittest.main()
