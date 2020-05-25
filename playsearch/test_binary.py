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
        ## TODO: write test
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

    def test_this_trick_lead_suit(self):
        binary_in = BinaryInput(np.zeros((6, 298)))

        this_trick = np.zeros((6, 3, 32))

        # sample 0: nothing was played in the trick yet
        # sample 1: rho led the spade A
        this_trick[1, 2] = get_binary_hand_from_cards([0])
        # sample 2: partner led the diamond K and rho put a small spade
        this_trick[2, 1] = get_binary_hand_from_cards([17])
        this_trick[2, 2] = get_binary_hand_from_cards([7])
        # sample 3: lho led a small club, partner followed small and rho discarded a small diamond
        this_trick[3, 0] = get_binary_hand_from_cards([31])
        this_trick[3, 1] = get_binary_hand_from_cards([31])
        this_trick[3, 2] = get_binary_hand_from_cards([23])
        # sample 4: lho led a small heart, partner discarded a small spade and rho discarded a small club
        this_trick[4, 0] = get_binary_hand_from_cards([15])
        this_trick[4, 1] = get_binary_hand_from_cards([7])
        this_trick[4, 2] = get_binary_hand_from_cards([31])
        # sample 5: lho led the club A, everyone followed small
        this_trick[5, 0] = get_binary_hand_from_cards([24])
        this_trick[5, 1] = get_binary_hand_from_cards([31])
        this_trick[5, 2] = get_binary_hand_from_cards([31])



        expected_lead_suit = np.array([
            [0, 0, 0, 0],   # nothing was lead. we are on lead
            [1, 0, 0, 0],   # spade was lead
            [0, 0, 1, 0],   # diamond was lead
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        binary_in.set_this_trick(this_trick)

        got_lead_suit = binary_in.get_this_trick_lead_suit()

        self.assertTrue(got_lead_suit.shape == (6, 4))
        self.assertTrue((expected_lead_suit == got_lead_suit).all())


if __name__ == '__main__':
    unittest.main()
