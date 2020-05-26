import numpy as np

from rollout import Rollout, PlayerRollout
from player import BatchPlayer, BatchPlayerLefty, follow_suit
from binary import *

## we just want to do the rollout by hand writing the instructions one by one
# this will be a very long script, but from here we get the ideas how to make the code nicer

def load_players():
    lefty = BatchPlayerLefty('lefty', '../notebooks/lefty_model/lefty-1000000')
    dummy = BatchPlayer('dummy', '../notebooks/dummy_model/dummy-920000')
    righty = BatchPlayer('righty', '../notebooks/righty_model/righty-1000000')
    decl = BatchPlayer('decl', '../notebooks/decl_model/decl-1000000')

    return [lefty, dummy, righty, decl]


def init_rollout(n_samples, players):
    this_trick = np.zeros(3 * 32)
    this_trick[2 * 32 + 17] = 1

    rollout = Rollout(
        n_samples = n_samples,
        players = players,
        on_play_i = 1,
        public_i = 1,
        public_hand = np.array([  # KQxx x A98x K8xx
            [0, 1, 1, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 2],
        ]).reshape(32),
        on_play_hand = np.array([ # AJ8xx J9x xx AJT
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

    return rollout


def main():
    players = load_players()

    n_samples = 32
    rollout = init_rollout(n_samples, players)

    s_all = np.array([True] * n_samples)  # boolean array used to select all samples

    rolls = rollout.player_rollouts
    [lefty_roll, dummy_roll, righty_roll, decl_roll] = rolls

    # now we step through the play rolling out

    # trick 1

    card_1 = np.zeros(n_samples, dtype=np.int) + 17  # the lead is always the DK
    trick = np.zeros((n_samples, 3*32))
    trick[s_all, 2*32 + card_1] = 1   # setting dummy's rho card to card_1
    rolls[1].x_in[:,0,192:288] = trick    # setting the current trick
    # now predict the next card
    card_2_softmax = follow_suit(
        players[1].next_cards_softmax(rolls[1].x_in[:,:1,:]),
        BinaryInput(rolls[1].x_in[:,0,:]).get_player_hand(),
        BinaryInput(rolls[1].x_in[:,0,:]).get_this_trick_lead_suit(),
    )
    card_2 = np.argmax(card_2_softmax, axis=1)
    print('card played by player 1')
    print(np.vstack([card_2, card_2_softmax[s_all, card_2]]).T)
    # we update the public hand of righty and decl (reflecting that card_2 was played)
    hand = rolls[2].get_public_hand(0)
    hand[s_all, card_2] -= 1
    rolls[2].set_public_hand(0, hand)
    hand = rolls[3].get_public_hand(0)
    hand[s_all, card_2] -= 1
    rolls[3].set_public_hand(0, hand)
    # we update the current trick for righty (reflecting that card_1 and card_2 have been played)
    trick = np.zeros((n_samples, 3*32))
    trick[s_all, 1*32 + card_1] = 1 # partner card
    trick[s_all, 2*32 + card_2] = 1 # rho card
    rolls[2].x_in[:,0,192:288] = trick
    # predict the card played by righty
    card_3_softmax = follow_suit(
        players[2].next_cards_softmax(rolls[2].x_in[:,:1,:]),
        BinaryInput(rolls[2].x_in[:,0,:]).get_player_hand(),
        BinaryInput(rolls[2].x_in[:,0,:]).get_this_trick_lead_suit(),
    )
    card_3 = np.argmax(card_3_softmax, axis=1)
    print('card played by player 2')
    print(np.vstack([card_3, card_3_softmax[s_all, card_3]]).T)
    # we update the trick for declarer reflecting that (card_1, card_2 and card_3 have been played)
    trick = np.zeros((n_samples, 3*32))
    trick[s_all, 0*32 + card_1] = 1 # lho
    trick[s_all, 1*32 + card_2] = 1 # partner
    trick[s_all, 2*32 + card_3] = 1 # rho
    rolls[3].x_in[:,0,192:288] = trick
    card_4_softmax = follow_suit(
        players[3].next_cards_softmax(rolls[3].x_in[:,:1,:]),
        BinaryInput(rolls[3].x_in[:,0,:]).get_player_hand(),
        BinaryInput(rolls[3].x_in[:,0,:]).get_this_trick_lead_suit(),
    )
    card_4 = np.argmax(card_4_softmax, axis=1)
    print('card played by player 3')
    print(np.vstack([card_4, card_4_softmax[s_all, card_4]]).T)
    # trick 1 complete
    print('trick 1 is complete')
    print(np.vstack([card_1, card_2, card_3, card_4]).T)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
