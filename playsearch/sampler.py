import numpy as np


class RandomSampler:

    def __init__(self, n_cards):
        self.shuffler = np.vectorize(np.random.permutation, signature='(n)->(n)')
        self.n_cards = n_cards

    def sample(self, n_samples, cards, split_i):
        x = np.zeros((n_samples, self.n_cards), np.int)
        x[:, :] = cards

        x_shuffled = self.shuffler(x)

        cards_1 = x_shuffled[:,:split_i]
        cards_2 = x_shuffled[:,split_i:]

        return distribute_cards(cards_1), distribute_cards(cards_2)


def distribute_cards(cards):
    n_samples, n_cards = cards.shape

    hand = np.zeros((n_samples, 32))
    row_indexes = np.arange(n_samples)

    for k in range(n_cards):
        hand[row_indexes, cards[:,k]] += 1

    return hand



