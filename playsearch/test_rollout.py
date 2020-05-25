import unittest
import numpy as np

from binary import BinaryInput
from player import BatchPlayer, BatchPlayerLefty
from rollout import RandomPlayer, PlayerRollout, Rollout, get_all_hidden_cards


class PlayerTest(unittest.TestCase):

    def test_prediction_shape(self):
        player = RandomPlayer(32)

        x = np.random.random((100, 10, 64))
        predictions = player.next_cards_softmax(x)
        self.assertTrue(predictions.shape == (100, 32))
        # prediction must sum to 1
        self.assertTrue(np.abs(np.min(np.sum(predictions, axis=1)) - 1) < 1e-6)
        self.assertTrue(np.abs(np.max(np.sum(predictions, axis=1)) - 1) < 1e-6)


class BatchPlayerTest(unittest.TestCase):

    def test_prediction_shape(self):
        lefty = BatchPlayerLefty('lefty', '../notebooks/lefty_model/lefty-1000000')
        dummy = BatchPlayer('dummy', '../notebooks/dummy_model/dummy-920000')
        righty = BatchPlayer('righty', '../notebooks/righty_model/righty-1000000')
        decl = BatchPlayer('decl', '../notebooks/decl_model/decl-1000000')
        
        # check that the graphs and sessions of the four different players are different
        self.assertFalse(lefty.sess is dummy.sess)
        self.assertFalse(lefty.sess is righty.sess)
        self.assertFalse(lefty.sess is decl.sess)
        self.assertFalse(dummy.sess is righty.sess)
        self.assertFalse(dummy.sess is decl.sess)
        self.assertFalse(righty.sess is decl.sess)

        self.assertFalse(lefty.graph is dummy.graph)
        self.assertFalse(lefty.graph is righty.graph)
        self.assertFalse(lefty.graph is decl.graph)
        self.assertFalse(dummy.graph is righty.graph)
        self.assertFalse(dummy.graph is decl.graph)
        self.assertFalse(righty.graph is decl.graph)

        players = [lefty, dummy, righty, decl]

        for player in players:
            x = np.random.random((100, 10, 298))
            predictions = player.next_cards_softmax(x)
            self.assertTrue(predictions.shape == (100, 32))
            # prediction must sum to 1
            self.assertTrue(np.abs(np.min(np.sum(predictions, axis=1)) - 1) < 1e-6)
            self.assertTrue(np.abs(np.max(np.sum(predictions, axis=1)) - 1) < 1e-6)

        for player in players:
            player.close()

    def test_lefty_prediction(self):
        # the lefty model is special because it does not predict the card on the first trick
        # not taking this into account should raise an exception
        lefty_wrong = BatchPlayer('lefty', '../notebooks/lefty_model/lefty-1000000')
        x = np.random.random((100, 10, 298))
        with self.assertRaises(ValueError):
            lefty_wrong.next_cards_softmax(x)
        lefty_wrong.close()


class PlayerRolloutTest(unittest.TestCase):

    def test_constructor(self):
        x_initial = np.ones((100, 298))
        player_rollout = PlayerRollout(player=None, x_initial=x_initial, n_tricks=3)

        self.assertTrue(player_rollout.n_samples == 100)
        self.assertTrue(player_rollout.n_ftrs == 298)
        self.assertTrue(player_rollout.n_tricks == 3)

        self.assertTrue((player_rollout.x_in[:,0,:] == x_initial).all())
        self.assertTrue((player_rollout.x_in[:,1,:] == np.zeros_like(x_initial)).all())
        self.assertTrue((player_rollout.x_in[:,2,:] == np.zeros_like(x_initial)).all())

        with self.assertRaises(IndexError):
            player_rollout.x_in[:,3,:]

    def test_get_set_player_hand(self):
        x_initial = np.zeros((2, 298))
        player_rollout = PlayerRollout(player=None, x_initial=x_initial, n_tricks=3)

        hand = np.zeros((2, 32))
        hand[0, [1,2,3,4,5,6,7,8,9,10,11,12,13]] = 1
        hand[1, [11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = 1

        player_rollout.set_player_hand(0, hand)
        self.assertTrue((player_rollout.get_player_hand(trick_i=0) == hand).all())
        self.assertTrue((player_rollout.get_player_hand(trick_i=1) == np.zeros_like(hand)).all())

        # mismatching sample sizes
        player_rollout = PlayerRollout(player=None, x_initial=np.zeros((100, 298)), n_tricks=3)
        with self.assertRaises(ValueError):
            player_rollout.set_player_hand(0, hand)

    def test_player_card_played(self):
        x_initial = np.zeros((2, 298))
        player_rollout = PlayerRollout(player=None, x_initial=x_initial, n_tricks=3)

        hand = np.zeros((2, 32))
        hand[0, [1,2,3,4,5,6,7,8,9,10,11,12,13]] = 1
        hand[1, [11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = 1

        player_rollout.set_player_hand(1, hand)

        card = np.zeros((2, 32))
        card[0,3] = 1
        card[1,15] = 1
        player_rollout.player_card_played(1, card)

        self.assertTrue((player_rollout.get_player_hand(1) == hand - card).all())
        self.assertTrue((player_rollout.get_public_hand(1) == np.zeros_like(hand)).all())
        self.assertTrue((player_rollout.get_player_hand(2) == np.zeros_like(hand)).all())

    def test_get_all_hidden_cards(self):
        visible_cards_list = [
            np.array([]),
            np.array([0]),
            np.array([7, 23, 23, 31, 31, 31]),
            np.array([0, 9, 18, 27, 7, 23, 23, 31, 31, 31]),
        ]

        hidden_cards_list = [
            np.array([0, 1, 2, 3, 4, 5, 6] + [7]*6 + [8, 9, 10, 11, 12, 13, 14] + [15] * 6 + [16, 17, 18, 19, 20, 21, 22] + [23] * 6 + [24, 25, 26, 27, 28, 29, 30] + [31] * 6),
            np.array([1, 2, 3, 4, 5, 6] + [7]*6 + [8, 9, 10, 11, 12, 13, 14] + [15] * 6 + [16, 17, 18, 19, 20, 21, 22] + [23] * 6 + [24, 25, 26, 27, 28, 29, 30] + [31] * 6),
            np.array([0, 1, 2, 3, 4, 5, 6] + [7]*5 + [8, 9, 10, 11, 12, 13, 14] + [15] * 6 + [16, 17, 18, 19, 20, 21, 22] + [23] * 4 + [24, 25, 26, 27, 28, 29, 30] + [31] * 3),
            np.array([1, 2, 3, 4, 5, 6] + [7]*5 + [8, 10, 11, 12, 13, 14] + [15] * 6 + [16, 17, 19, 20, 21, 22] + [23] * 4 + [24, 25, 26, 28, 29, 30] + [31] * 3),
        ]

        for visible_cards, hidden_cards in zip(visible_cards_list, hidden_cards_list):
            self.assertTrue((get_all_hidden_cards(visible_cards) == hidden_cards).all())


class RolloutTest(unittest.TestCase):

    def test_rollout_init(self):

        this_trick = np.zeros(3 * 32)
        this_trick[2 * 32 + 17] = 1

        n_samples = 8  # something strange happens when this is > 1000.   TODO: investigate

        rollout = Rollout(
            n_samples = n_samples,
            players = [RandomPlayer(32) for _ in range(4)],
            on_play_i = 1,
            public_i = 1,
            public_hand = np.array([
                [0, 1, 1, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1, 2],
            ]).reshape(32),
            on_play_hand = np.array([
                [1, 0, 0, 1, 0, 0, 1, 2],
                [0, 0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 2],
                [1, 0, 0, 1, 1, 0, 0, 0],
            ]).reshape(32),
            n_cards_hidden1 = 12,
            n_cards_hidden2 = 13,
            cards_considered = [16, 21, 23],
            level = 4,
            strain = np.array([0, 1, 0, 0, 0]),
            trick_i = 0,
            last_trick_lead_i = 0,
            last_trick = np.zeros(4 * 32),
            this_trick_lead_i = 0,
            this_trick = this_trick
        )

        self.assertTrue(rollout.cards_considered == [16, 21, 23])

        self.assertTrue(len(rollout.player_rollouts) == 4)

        for player_rollout in rollout.player_rollouts:
            self.assertTrue(player_rollout.x_in.shape == (n_samples, 13, 298))

        for i, player_rollout in enumerate(rollout.player_rollouts):
            sum_player_hand_samples = np.sum(player_rollout.get_player_hand(0), axis=0).reshape((4, 8))
            sum_public_hand_samples = np.sum(player_rollout.get_public_hand(0), axis=0).reshape((4, 8))

            # check that there are no common cards between the player hand and the public hand (except the smallest card that is reused)
            self.assertTrue((((sum_player_hand_samples / n_samples) * (sum_public_hand_samples / n_samples))[:,:7] == np.zeros((4, 7))).all())
            self.assertTrue(np.sum(sum_public_hand_samples) / n_samples == 13)
            if i == 0:
                self.assertTrue(np.sum(sum_player_hand_samples) / n_samples == 12)
            else:
                self.assertTrue(np.sum(sum_player_hand_samples) / n_samples == 13)

        [lefty, dummy, righty, declarer] = rollout.player_rollouts

        player_hand_sum = lefty.get_player_hand(0) + dummy.get_player_hand(0) + righty.get_player_hand(0) + declarer.get_player_hand(0)

        all_cards = np.array([
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  6.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  6.],
            [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  6.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  6.]
        ]).reshape(32)
        self.assertTrue((player_hand_sum == all_cards).all())

        dummy_sum = lefty.get_public_hand(0) + righty.get_public_hand(0) + declarer.get_public_hand(0)
        self.assertTrue((dummy_sum == 3*rollout.public_hand).all())

        self.assertTrue((lefty.get_public_hand(0) == rollout.public_hand).all())
        self.assertTrue((righty.get_public_hand(0) == rollout.public_hand).all())
        self.assertTrue((declarer.get_public_hand(0) == rollout.public_hand).all())

        self.assertTrue((declarer.get_public_hand(0) == dummy.get_player_hand(0)).all())

        for player_rollout in rollout.player_rollouts:
            self.assertTrue((BinaryInput(player_rollout.x_in[:,0,:]).get_level() == 4).all())
            self.assertTrue((BinaryInput(player_rollout.x_in[:,0,:]).get_strain() == np.array([0,1,0,0,0])).all())
            self.assertTrue((BinaryInput(player_rollout.x_in[:,0,:]).get_last_trick_lead() == 0).all())
            self.assertTrue((BinaryInput(player_rollout.x_in[:,0,:]).get_last_trick() == np.zeros((n_samples, 4, 32))).all())
            self.assertTrue((BinaryInput(player_rollout.x_in[:,0,:]).get_this_trick() == np.zeros((n_samples, 3, 32))).all())


if __name__ == '__main__':
    unittest.main()
