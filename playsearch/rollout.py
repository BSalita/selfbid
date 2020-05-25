import numpy as np

from binary import BinaryInput, get_cards_from_binary_hand, get_binary_hand_from_cards
from sampler import RandomSampler

class RandomPlayer:

    def __init__(self, n_dim):
        self.n_dim = n_dim

    def next_cards_softmax(self, x):
        n_samples, n_steps, n_ftrs = x.shape

        # makes a random output, later will use NN

        p = np.random.random((n_samples, self.n_dim))
        p_sum = np.sum(p, axis=1, keepdims=True)
        assert(p_sum.shape == (n_samples, 1))

        p_norm = p / p_sum

        return p_norm


class PlayerRollout:

    def __init__(self, player, x_initial, n_tricks):
        self.n_samples, self.n_ftrs = x_initial.shape
        self.n_tricks = n_tricks

        # initialize the inputs and outputs
        self.x_in = np.zeros((self.n_samples, self.n_tricks, self.n_ftrs), np.float16)
        self.x_in[:,0,:] = x_initial

    def set_player_hand(self, trick_i, hand):
        BinaryInput(self.x_in[:,trick_i,:]).set_player_hand(hand)

    def get_player_hand(self, trick_i):
        return BinaryInput(self.x_in[:,trick_i,:]).get_player_hand()

    def set_public_hand(self, trick_i, hand):
        BinaryInput(self.x_in[:,trick_i,:]).set_public_hand(hand)

    def get_public_hand(self, trick_i):
        return BinaryInput(self.x_in[:,trick_i,:]).get_public_hand()

    def player_card_played(self, trick_i, card_one_hot):
        binary_in = BinaryInput(self.x_in[:,trick_i,:])
        binary_in.set_player_hand(binary_in.get_player_hand() - card_one_hot)

    def public_card_played(self, trick_i, card_one_hot):
        binary_in = BinaryInput(self.x_in[:,trick_i,:])
        binary_in.set_public_hand(binary_in.get_public_hand() - card_one_hot)

    def get_next_cards_softmax(self, trick_i):
        # should this be: player.next_cards_softmax(self.x_in[:, : trick_i,:])  ?
        # (because we have to give the whole sequence)
        return player.next_cards_softmax(self.x_in[:,trick_i,:])


# rollout is done from the point of view of a player.
# the player cards and dummy cards are observed
# hidden cards are sampled
# the purpose of a rollout is to evaluate a possible play by simulating the outcome on a sample of possible hands
# state of play at the point of doing a rollout:
#   - player hand (stays constant in all samples)
#   - dummy hand (stays constant in all samples)
#   - level and strain of contract
#   - previous trick and who was on lead
#   - current trick so far, who is on play now
#   - play to evaluate. the next card will be the card to evaluate and the following cards will be played according to the model
#   - how many tricks do we need to roll out?

class Rollout:

    def __init__(self, n_samples, players, on_play_i, public_i, on_play_hand, public_hand, n_cards_hidden1, n_cards_hidden2, cards_considered, level, strain, trick_i, last_trick_lead_i, last_trick, this_trick_lead_i, this_trick):
        self.n_samples = n_samples
        self.players = players
        self.on_play_i = on_play_i
        self.public_i = public_i
        self.on_play_hand = on_play_hand
        self.public_hand = public_hand
        self.n_cards_hidden1 = n_cards_hidden1
        self.n_cards_hidden2 = n_cards_hidden2
        self.cards_considered = cards_considered
        self.level = level
        self.strain = strain
        self.trick_i = trick_i
        self.n_tricks_left = 13 - self.trick_i
        self.last_trick_lead_i = last_trick_lead_i
        self.last_trick = last_trick
        self.this_trick_lead_i = this_trick_lead_i
        self.this_trick = this_trick

        self.player_rollouts = self.init_player_rollouts()

    def init_sampled_hands(self):
        visible_cards = np.concatenate(
            list(map(get_cards_from_binary_hand, [self.on_play_hand, self.public_hand])) + \
            list(map(get_cards_from_binary_hand, [self.this_trick[:32], self.this_trick[32:64], self.this_trick[64:]]))
        )

        hidden_cards = get_all_hidden_cards(visible_cards)

        random_sampler = RandomSampler(len(hidden_cards))
        split_i = self.n_cards_hidden1

        hand1, hand2 = random_sampler.sample(self.n_samples, hidden_cards, split_i)
        hidden_hand_samples = [hand1, hand2]

        on_play_hand = np.zeros((self.n_samples, 32), np.float16)
        on_play_hand[:,:] = self.on_play_hand
        public_hand = np.zeros((self.n_samples, 32), np.float16)
        public_hand[:,:] = self.public_hand

        if self.on_play_i == self.public_i: # dummy is on play
            sampled_hands = [hand1, public_hand, hand2, on_play_hand]  # [lefty, dummy, righty, declarer]
        else:
            sampled_hands = []  # [lefty, dummy, righty, declarer]
            k = 0
            for i in range(4):
                if i == self.on_play_i:
                    sampled_hands.append(on_play_hand)
                elif i == self.public_i:
                    sampled_hands.append(public_hand)
                else:
                    sampled_hands.append(hidden_hand_samples[k])
                    k += 1

        return sampled_hands

    def init_x_input(self):
        x = np.zeros((self.n_samples, 298), np.float16)
        binary_in = BinaryInput(x)

        binary_in.set_level(self.level)
        binary_in.set_strain(self.strain)
        binary_in.set_last_trick_lead(np.zeros(self.n_samples) + self.last_trick_lead_i)

        last_trick = binary_in.get_last_trick().reshape((-1, 4*32))
        last_trick[:, :] = self.last_trick
        binary_in.set_last_trick(last_trick)

        return binary_in.x

    def init_player_rollouts(self):
        sampled_hands = self.init_sampled_hands()

        x_in = self.init_x_input()

        player_rollouts = [PlayerRollout(player, x_in, self.n_tricks_left) for player in self.players]

        for i, hand in enumerate(sampled_hands):
            if i == self.public_i:
                player_rollouts[i].set_player_hand(0, hand)
                player_rollouts[i].set_public_hand(0, sampled_hands[(i + 2) % 4])
            else:
                player_rollouts[i].set_player_hand(0, hand)
                player_rollouts[i].set_public_hand(0, sampled_hands[self.public_i])

        # current trick is still missing at this point
        # current trick is always in the order: lho, pard, rho

        return player_rollouts

    def rollout(self):
        pass


def get_all_hidden_cards(visible_cards):
    all_cards_hand = np.array([
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
    ]).reshape(32)

    return get_cards_from_binary_hand(all_cards_hand - get_binary_hand_from_cards(visible_cards))
