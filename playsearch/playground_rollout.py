import numpy as np

from rollout import Rollout, PlayerRollout
from player import BatchPlayer, BatchPlayerLefty, follow_suit, get_trick_winner_i
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

    import time
    start_t = time.time()

    n_samples = 16
    strain = np.zeros((n_samples, 5))
    strain[:,1] = 1 # we set spades trumps. TODO: how to avoid repeat-hardcoding this here?
    rollout = init_rollout(n_samples, players)

    s_all = np.array([True] * n_samples)  # boolean array used to select all samples

    rolls = rollout.player_rollouts
    [lefty_roll, dummy_roll, righty_roll, decl_roll] = rolls

    # now we step through the play rolling out

    ### TRICK 1

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
    hand = rolls[2].get_public_hand(0).copy()
    hand[s_all, card_2] -= 1
    rolls[2].set_public_hand(0, hand)
    hand = rolls[3].get_public_hand(0).copy()
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
    # creating the one-hot-encoded full trick
    full_trick_1 = np.zeros((n_samples, 4 * 32))
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_1] = 1
    full_trick_1[:,0*32:(0+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_2] = 1
    full_trick_1[:,1*32:(1+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_3] = 1
    full_trick_1[:,2*32:(2+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_4] = 1
    full_trick_1[:,3*32:(3+1)*32] = card_i
    winner_trick_1 = get_trick_winner_i(full_trick_1, strain)
    print(np.hstack([card_1.reshape((-1, 1)), card_2.reshape((-1, 1)), card_3.reshape((-1, 1)), card_4.reshape(-1, 1), winner_trick_1]))

    ## preparing the input for trick 2
    # setting all player hands
    rolls[0].set_player_hand(1, rolls[0].get_player_hand(0))
    hand = rolls[1].get_player_hand(0).copy()
    hand[s_all, card_2] -= 1
    rolls[1].set_player_hand(1, hand)
    hand = rolls[2].get_player_hand(0).copy()
    hand[s_all, card_3] -= 1
    rolls[2].set_player_hand(1, hand)
    hand = rolls[3].get_player_hand(0).copy()
    hand[s_all, card_4] -= 1
    rolls[3].set_player_hand(1, hand)

    assert np.sum(rolls[0].get_player_hand(0)[0]) == 12
    assert np.sum(rolls[0].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[1].get_player_hand(0)[0]) == 13
    assert np.sum(rolls[1].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[2].get_player_hand(0)[0]) == 13
    assert np.sum(rolls[2].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[3].get_player_hand(0)[0]) == 13
    assert np.sum(rolls[3].get_player_hand(1)[0]) == 12

    # setting all public hands
    rolls[0].set_public_hand(1, rolls[1].get_player_hand(1))
    rolls[1].set_public_hand(1, rolls[3].get_player_hand(1))
    rolls[2].set_public_hand(1, rolls[1].get_player_hand(1))
    rolls[3].set_public_hand(1, rolls[1].get_player_hand(1))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,1,64:192] = full_trick_1
        rolls[i].x_in[:,1,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,1,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[:,1,288] = 1
        rolls[i].x_in[:,1,292] = 4  # level
        rolls[i].x_in[:,1,293:] = strain


    ### TRICK 2
    print('\n\n ==== TRICK 2 \n\n')

    # we assume that it's dummy's turn as he won by the DA (this is not always true. occasionally righty ruffed)
    card_1_softmax = follow_suit(
        players[1].next_cards_softmax(rolls[1].x_in[:,:2,:]),
        BinaryInput(rolls[1].x_in[:,1,:]).get_player_hand(),
        BinaryInput(rolls[1].x_in[:,1,:]).get_this_trick_lead_suit(),
    )
    card_1 = np.argmax(card_1_softmax, axis=1)
    print('card played by player 1')
    print(np.vstack([card_1, card_1_softmax[s_all, card_1]]).T)
    # we update the public hand of all players who are yet to play to the trick
    hand = rolls[2].get_public_hand(1).copy()
    hand[s_all, card_1] -= 1
    rolls[2].set_public_hand(1, hand)
    rolls[3].set_public_hand(1, hand)
    rolls[0].set_public_hand(1, hand)
    # we update the current trick for all players who are yet to play to the trick
    rolls[2].x_in[s_all, 1, 192 + 2*32 + card_1] = 1
    rolls[3].x_in[s_all, 1, 192 + 1*32 + card_1] = 1
    rolls[0].x_in[s_all, 1, 192 + 0*32 + card_1] = 1
    # we predict righty's card
    card_2_softmax = follow_suit(
        players[2].next_cards_softmax(rolls[2].x_in[:,:2,:]),
        BinaryInput(rolls[2].x_in[:,1,:]).get_player_hand(),
        BinaryInput(rolls[2].x_in[:,1,:]).get_this_trick_lead_suit(),
    )
    card_2 = np.argmax(card_2_softmax, axis=1)
    print('card played by player 2')
    print(np.vstack([card_2, card_2_softmax[s_all, card_2]]).T)
    # we update the current trick for all players who are yet to play to the trick
    rolls[3].x_in[s_all, 1, 192 + 2*32 + card_2] = 1
    rolls[0].x_in[s_all, 1, 192 + 1*32 + card_2] = 1
    # we predict declarer's card
    card_3_softmax = follow_suit(
        players[3].next_cards_softmax(rolls[3].x_in[:,:2,:]),
        BinaryInput(rolls[3].x_in[:,1,:]).get_player_hand(),
        BinaryInput(rolls[3].x_in[:,1,:]).get_this_trick_lead_suit(),
    )
    card_3 = np.argmax(card_3_softmax, axis=1)
    print('card played by player 3')
    print(np.vstack([card_3, card_3_softmax[s_all, card_3]]).T)
    # we update the current trick for all players who are yet to play to the trick
    rolls[0].x_in[s_all, 1, 192 + 2*32 + card_3] = 1
    # we predict lefty's card
    card_4_softmax = follow_suit(
        players[0].next_cards_softmax(rolls[0].x_in[:,:2,:]),
        BinaryInput(rolls[0].x_in[:,1,:]).get_player_hand(),
        BinaryInput(rolls[0].x_in[:,1,:]).get_this_trick_lead_suit(),
    )
    card_4 = np.argmax(card_4_softmax, axis=1)
    print('card played by player 0')
    print(np.vstack([card_4, card_4_softmax[s_all, card_4]]).T)
    # trick 2 complete
    print('trick 2 is complete')
    # creating the one-hot-encoded full trick
    full_trick_2 = np.zeros((n_samples, 4 * 32))
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_1] = 1
    full_trick_2[:,0*32:(0+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_2] = 1
    full_trick_2[:,1*32:(1+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_3] = 1
    full_trick_2[:,2*32:(2+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_4] = 1
    full_trick_2[:,3*32:(3+1)*32] = card_i
    winner_trick_2 = get_trick_winner_i(full_trick_2, strain)
    print(np.hstack([card_1.reshape((-1, 1)), card_2.reshape((-1, 1)), card_3.reshape((-1, 1)), card_4.reshape(-1, 1), winner_trick_2]))

    ## preparing the input for trick 3
    # setting all player hands
    hand = rolls[0].get_player_hand(1).copy()
    hand[s_all, card_4] -= 1
    rolls[0].set_player_hand(2, hand)
    hand = rolls[1].get_player_hand(1).copy()
    hand[s_all, card_1] -= 1
    rolls[1].set_player_hand(2, hand)
    hand = rolls[2].get_player_hand(1).copy()
    hand[s_all, card_2] -= 1
    rolls[2].set_player_hand(2, hand)
    hand = rolls[3].get_player_hand(1).copy()
    hand[s_all, card_3] -= 1
    rolls[3].set_player_hand(2, hand)

    assert np.sum(rolls[0].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[0].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[1].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[1].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[2].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[2].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[3].get_player_hand(1)[0]) == 12
    assert np.sum(rolls[3].get_player_hand(2)[0]) == 11

    # setting all public hands
    rolls[0].set_public_hand(2, rolls[1].get_player_hand(2))
    rolls[1].set_public_hand(2, rolls[3].get_player_hand(2))
    rolls[2].set_public_hand(2, rolls[1].get_player_hand(2))
    rolls[3].set_public_hand(2, rolls[1].get_player_hand(2))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,2,64:192] = full_trick_2
        rolls[i].x_in[:,2,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,2,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[:,2,289] = 1  # last trick lead player index
        rolls[i].x_in[:,2,292] = 4  # level
        rolls[i].x_in[:,2,293:] = strain

    ### TRICK 3
    print('\n\n ==== TRICK 3 \n\n')

    # we assume that it's declarer's turn as he won by the SA (this is not always true)
    card_1_softmax = follow_suit(
        players[3].next_cards_softmax(rolls[3].x_in[:,:3,:]),
        BinaryInput(rolls[3].x_in[:,2,:]).get_player_hand(),
        BinaryInput(rolls[3].x_in[:,2,:]).get_this_trick_lead_suit(),
    )
    card_1 = np.argmax(card_1_softmax, axis=1)
    print('card played by player 3')
    print(np.vstack([card_1, card_1_softmax[s_all, card_1]]).T)
    # we update the current trick for all players who are yet to play to the trick
    rolls[0].x_in[s_all, 2, 192 + 2*32 + card_1] = 1
    rolls[1].x_in[s_all, 2, 192 + 1*32 + card_1] = 1
    rolls[2].x_in[s_all, 2, 192 + 0*32 + card_1] = 1
    # we update the public hand of dummy
    hand = rolls[1].get_public_hand(2).copy()
    hand[s_all, card_1] -= 1
    rolls[1].set_public_hand(2, hand)
    # we predict lefty's card
    card_2_softmax = follow_suit(
        players[0].next_cards_softmax(rolls[0].x_in[:,:3,:]),
        BinaryInput(rolls[0].x_in[:,2,:]).get_player_hand(),
        BinaryInput(rolls[0].x_in[:,2,:]).get_this_trick_lead_suit(),
    )
    card_2 = np.argmax(card_2_softmax, axis=1)
    print('card played by player 0')
    print(np.vstack([card_2, card_2_softmax[s_all, card_2]]).T)
    # we update the current trick for all players who are yet to play to the trick
    rolls[1].x_in[s_all, 2, 192 + 2*32 + card_2] = 1
    rolls[2].x_in[s_all, 2, 192 + 1*32 + card_2] = 1
    # we predict dummy's card
    card_3_softmax = follow_suit(
        players[1].next_cards_softmax(rolls[1].x_in[:,:3,:]),
        BinaryInput(rolls[1].x_in[:,2,:]).get_player_hand(),
        BinaryInput(rolls[1].x_in[:,2,:]).get_this_trick_lead_suit(),
    )
    card_3 = np.argmax(card_3_softmax, axis=1)
    print('card played by player 1')
    print(np.vstack([card_3, card_3_softmax[s_all, card_3]]).T)
    # we update the current trick for all players who are yet to play to the trick
    rolls[2].x_in[s_all, 2, 192 + 2*32 + card_3] = 1
    # we update the public hand of players yet to play to the trick
    hand = rolls[2].get_public_hand(2).copy()
    hand[s_all, card_3] -= 1
    rolls[2].set_public_hand(2, hand)
    # we predict righty's card
    card_4_softmax = follow_suit(
        players[2].next_cards_softmax(rolls[2].x_in[:,:3,:]),
        BinaryInput(rolls[2].x_in[:,2,:]).get_player_hand(),
        BinaryInput(rolls[2].x_in[:,2,:]).get_this_trick_lead_suit(),
    )
    card_4 = np.argmax(card_4_softmax, axis=1)
    print('card played by player 2')
    print(np.vstack([card_4, card_4_softmax[s_all, card_4]]).T)
    # trick 3 complete
    print('trick 3 is complete')
    # creating the one-hot-encoded full trick
    full_trick_3 = np.zeros((n_samples, 4 * 32))
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_1] = 1
    full_trick_3[:,0*32:(0+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_2] = 1
    full_trick_3[:,1*32:(1+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_3] = 1
    full_trick_3[:,2*32:(2+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_4] = 1
    full_trick_3[:,3*32:(3+1)*32] = card_i
    winner_trick_3 = get_trick_winner_i(full_trick_3, strain)
    print(np.hstack([card_1.reshape((-1, 1)), card_2.reshape((-1, 1)), card_3.reshape((-1, 1)), card_4.reshape(-1, 1), winner_trick_3]))

    ## preparing the input for trick 4
    # setting all player hands
    hand = rolls[0].get_player_hand(2).copy()
    hand[s_all, card_2] -= 1
    rolls[0].set_player_hand(3, hand)
    hand = rolls[1].get_player_hand(2).copy()
    hand[s_all, card_3] -= 1
    rolls[1].set_player_hand(3, hand)
    hand = rolls[2].get_player_hand(2).copy()
    hand[s_all, card_4] -= 1
    rolls[2].set_player_hand(3, hand)
    hand = rolls[3].get_player_hand(2).copy()
    hand[s_all, card_1] -= 1
    rolls[3].set_player_hand(3, hand)

    assert np.sum(rolls[0].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[0].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[1].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[1].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[2].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[2].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[3].get_player_hand(2)[0]) == 11
    assert np.sum(rolls[3].get_player_hand(3)[0]) == 10

    # setting all public hands
    rolls[0].set_public_hand(3, rolls[1].get_player_hand(3))
    rolls[1].set_public_hand(3, rolls[3].get_player_hand(3))
    rolls[2].set_public_hand(3, rolls[1].get_player_hand(3))
    rolls[3].set_public_hand(3, rolls[1].get_player_hand(3))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,2,64:192] = full_trick_3
        rolls[i].x_in[:,2,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,2,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[:,2,291] = 1  # last trick lead player index
        rolls[i].x_in[:,2,292] = 4  # level
        rolls[i].x_in[:,2,293:] = strain

    ### TRICK 4
    print('\n\n ==== TRICK 4 \n\n')

    # we assume that it's dummy's turn as he won by the SQ (this is not always true)
    card_1_softmax = follow_suit(
        players[1].next_cards_softmax(rolls[1].x_in[:,:4,:]),
        BinaryInput(rolls[1].x_in[:,3,:]).get_player_hand(),
        BinaryInput(rolls[1].x_in[:,3,:]).get_this_trick_lead_suit(),
    )
    card_1 = np.argmax(card_1_softmax, axis=1)
    print('card played by player 1')
    print(np.vstack([card_1, card_1_softmax[s_all, card_1]]).T)
    # we update the current trick for all players yet to play to the trick
    rolls[2].x_in[s_all, 3, 192 + 2*32 + card_1] = 1
    rolls[3].x_in[s_all, 3, 192 + 1*32 + card_1] = 1
    rolls[0].x_in[s_all, 3, 192 + 0*32 + card_1] = 1
    # we update the public hand for all players yet to play to the trick
    hand = rolls[2].get_public_hand(3).copy()
    hand[s_all, card_1] -= 1
    rolls[2].set_public_hand(3, hand)
    rolls[3].set_public_hand(3, hand)
    rolls[0].set_public_hand(3, hand)
    # we predict righty's card
    card_2_softmax = follow_suit(
        players[2].next_cards_softmax(rolls[2].x_in[:,:4,:]),
        BinaryInput(rolls[2].x_in[:,3,:]).get_player_hand(),
        BinaryInput(rolls[2].x_in[:,3,:]).get_this_trick_lead_suit(),
    )
    card_2 = np.argmax(card_2_softmax, axis=1)
    print('card played by player 2')
    print(np.vstack([card_2, card_2_softmax[s_all, card_2]]).T)
    # we update the current trick for all players yet to play to the trick
    rolls[3].x_in[s_all, 3, 192 + 2*32 + card_2] = 1
    rolls[0].x_in[s_all, 3, 192 + 1*32 + card_2] = 1
    # we predict declarer's card
    card_3_softmax = follow_suit(
        players[3].next_cards_softmax(rolls[3].x_in[:,:4,:]),
        BinaryInput(rolls[3].x_in[:,3,:]).get_player_hand(),
        BinaryInput(rolls[3].x_in[:,3,:]).get_this_trick_lead_suit(),
    )
    card_3 = np.argmax(card_3_softmax, axis=1)
    print('card played by player 3')
    print(np.vstack([card_3, card_3_softmax[s_all, card_3]]).T)
    # we update the current trick for all players yet to play to the trick
    rolls[0].x_in[s_all, 3, 192 + 2*32 + card_3] = 1
    # we predict lefty's card
    card_4_softmax = follow_suit(
        players[0].next_cards_softmax(rolls[0].x_in[:,:4,:]),
        BinaryInput(rolls[0].x_in[:,3,:]).get_player_hand(),
        BinaryInput(rolls[0].x_in[:,3,:]).get_this_trick_lead_suit(),
    )
    card_4 = np.argmax(card_4_softmax, axis=1)
    print('card played by player 0')
    print(np.vstack([card_4, card_4_softmax[s_all, card_4]]).T)
    # trick 4 complete
    print('trick 4 is complete')
    # creating the one-hot-encoded full trick
    full_trick_4 = np.zeros((n_samples, 4 * 32))
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_1] = 1
    full_trick_4[:,0*32:(0+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_2] = 1
    full_trick_4[:,1*32:(1+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_3] = 1
    full_trick_4[:,2*32:(2+1)*32] = card_i
    card_i = np.zeros((n_samples, 32))
    card_i[s_all, card_4] = 1
    full_trick_4[:,3*32:(3+1)*32] = card_i
    winner_trick_4 = get_trick_winner_i(full_trick_4, strain)
    print(np.hstack([card_1.reshape((-1, 1)), card_2.reshape((-1, 1)), card_3.reshape((-1, 1)), card_4.reshape(-1, 1), winner_trick_4]))

    ## preparing the input for trick 5
    # setting all player hands
    hand = rolls[0].get_player_hand(3).copy()
    hand[s_all, card_4] -= 1
    rolls[0].set_player_hand(4, hand)
    hand = rolls[1].get_player_hand(3).copy()
    hand[s_all, card_1] -= 1
    rolls[1].set_player_hand(4, hand)
    hand = rolls[2].get_player_hand(3).copy()
    hand[s_all, card_2] -= 1
    rolls[2].set_player_hand(4, hand)
    hand = rolls[3].get_player_hand(3).copy()
    hand[s_all, card_3] -= 1
    rolls[3].set_player_hand(4, hand)

    assert np.sum(rolls[0].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[0].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[1].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[1].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[2].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[2].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[3].get_player_hand(3)[0]) == 10
    assert np.sum(rolls[3].get_player_hand(4)[0]) == 9

    # setting all public hands
    rolls[0].set_public_hand(4, rolls[1].get_player_hand(4))
    rolls[1].set_public_hand(4, rolls[3].get_player_hand(4))
    rolls[2].set_public_hand(4, rolls[1].get_player_hand(4))
    rolls[3].set_public_hand(4, rolls[1].get_player_hand(4))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,3,64:192] = full_trick_4
        rolls[i].x_in[:,3,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,3,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[:,3,291] = 1  # last trick lead player index
        rolls[i].x_in[:,3,292] = 4  # level
        rolls[i].x_in[:,3,293:] = strain

    print('\n\n === TRICK 5 \n\n')

    ### latest by now the fun is over because it's not clear at all who will lead on the next trick
    ### we cannot make assumptions anymore and have to treat different cases differently
    ### the code gets a lot more complicated

    trick_i = 4

    # leader of trick 4. usually this will be dreived from winner_trick_3, but now we'll just hard code it to the dummy
    leader_trick_4 = np.zeros(n_samples, dtype=np.int) + 1
    leader_trick_5 = (leader_trick_4 + winner_trick_4.reshape(n_samples)) % 4
        # on_play = np.zeros((n_samples, 8), dtype=np.int)
        # on_play[:,0:1] =) leader_trick_5
    # for i in range(1, 8):
    #     on_play[:,i:(i+1)] = (on_play[:,(i-1):i] + 1) % 4

    n_cards_expected = 4 * np.ones(n_samples, dtype=np.int)
    n_cards_seen = np.zeros(n_samples, dtype=np.int)

    # we cycle through each player twice (lefty -> dummy -> righty -> declarer). 
    # we take a prediction, bit if it was not the player's turn, we don't update the game state
    trick_cards = -np.ones((n_samples, 4), dtype=np.int)
    for player_i in [0, 1, 2, 3, 0, 1, 2, 3]:
        if np.all(n_cards_seen >= n_cards_expected):
            break  # we are done with this trick. no need to cycle further

        s_on_play = ((n_cards_seen > 0) | (leader_trick_5 == player_i)) & (n_cards_seen < n_cards_expected)
        # we predict the next card
        card_softmax = follow_suit(
            players[player_i].next_cards_softmax(rolls[player_i].x_in[:,:(trick_i + 1),:]),
            BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_player_hand(),
            BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_this_trick_lead_suit(),
        )
        card = np.argmax(card_softmax, axis=1)

        # print('card played by player %i' % player_i)
        # print(np.vstack([card, card_softmax[s_all, card], s_on_play]).T)
        trick_cards[s_on_play, n_cards_seen[s_on_play]] = card[s_on_play]

        print(trick_cards)
            
        rolls[(player_i + 1) % 4].x_in[s_on_play & (n_cards_seen < 3), trick_i, 192 + 2*32 + card[s_on_play & (n_cards_seen < 3)]] = 1
        rolls[(player_i + 2) % 4].x_in[s_on_play & (n_cards_seen < 2), trick_i, 192 + 1*32 + card[s_on_play & (n_cards_seen < 2)]] = 1
        rolls[(player_i + 3) % 4].x_in[s_on_play & (n_cards_seen < 1), trick_i, 192 + 0*32 + card[s_on_play & (n_cards_seen < 1)]] = 1

        # we update the public hand for all players yet to play to the trick
        hand = rolls[player_i].get_player_hand(trick_i).copy()
        hand[s_on_play, card[s_on_play]] -= 1
        if player_i == 1: # dummy
            rolls[2].x_in[s_on_play & (n_cards_seen < 3), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 3)]
            rolls[3].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]
            rolls[0].x_in[s_on_play & (n_cards_seen < 1), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 1)]
        
        if player_i == 3: # declarer on play and dummy hasn't played yet (i.e. n_cards_seen < 2)
            rolls[1].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]

        n_cards_seen[s_on_play] += 1

    print("trick %s is complete" % trick_i)
    # finding the trick winners
    # creating the one-hot-encoded full trick
    full_trick = np.zeros((n_samples, 4 * 32))
    for i in range(4):
        full_trick[s_all, i*32 + trick_cards[:,i]] = 1

    
    winner_trick_5 = get_trick_winner_i(full_trick, strain)
    leader_trick_6 = (leader_trick_5 + winner_trick_5.reshape(-1)) % 4
    print(np.hstack([trick_cards, leader_trick_5.reshape((-1, 1)), leader_trick_6.reshape((-1, 1))]))

    ## preparing the input for trick 6
    # setting all player hands

    for i in range(4):
        hand = rolls[i].get_player_hand(trick_i).copy()
        player_i_card_index = (i - leader_trick_5) % 4
        card = trick_cards[s_all, player_i_card_index]
        hand[s_all, card] -= 1
        rolls[i].set_player_hand(trick_i + 1, hand)


    assert np.sum(rolls[0].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[0].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[1].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[1].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[2].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[2].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[3].get_player_hand(4)[0]) == 9
    assert np.sum(rolls[3].get_player_hand(5)[0]) == 8

    # setting all public hands
    rolls[0].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
    rolls[1].set_public_hand(trick_i+1, rolls[3].get_player_hand(trick_i))
    rolls[2].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
    rolls[3].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,trick_i+1,64:192] = full_trick
        rolls[i].x_in[:,trick_i+1,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,trick_i+1,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[s_all,trick_i+1,288 + leader_trick_5] = 1  # last trick lead player index
        rolls[i].x_in[:,trick_i+1,292] = 4  # level
        rolls[i].x_in[:,trick_i+1,293:] = strain


    print('\n\n === TRICK 6 \n\n')

    trick_i = trick_i + 1

    n_cards_expected = 4 * np.ones(n_samples, dtype=np.int)
    n_cards_seen = np.zeros(n_samples, dtype=np.int)

    # we cycle through each player twice (lefty -> dummy -> righty -> declarer). 
    # we take a prediction, bit if it was not the player's turn, we don't update the game state
    trick_cards = -np.ones((n_samples, 4), dtype=np.int)
    for player_i in [0, 1, 2, 3, 0, 1, 2, 3]:
        if np.all(n_cards_seen >= n_cards_expected):
            break  # we are done with this trick. no need to cycle further

        s_on_play = ((n_cards_seen > 0) | (leader_trick_6 == player_i)) & (n_cards_seen < n_cards_expected)
        # we predict the next card
        card_softmax = follow_suit(
            players[player_i].next_cards_softmax(rolls[player_i].x_in[:,:(trick_i + 1),:]),
            BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_player_hand(),
            BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_this_trick_lead_suit(),
        )
        card = np.argmax(card_softmax, axis=1)

        trick_cards[s_on_play, n_cards_seen[s_on_play]] = card[s_on_play]

        #print(trick_cards)
            
        # we update the current trick for all players yet to play to this trick
        rolls[(player_i + 1) % 4].x_in[s_on_play & (n_cards_seen < 3), trick_i, 192 + 2*32 + card[s_on_play & (n_cards_seen < 3)]] = 1
        rolls[(player_i + 2) % 4].x_in[s_on_play & (n_cards_seen < 2), trick_i, 192 + 1*32 + card[s_on_play & (n_cards_seen < 2)]] = 1
        rolls[(player_i + 3) % 4].x_in[s_on_play & (n_cards_seen < 1), trick_i, 192 + 0*32 + card[s_on_play & (n_cards_seen < 1)]] = 1

        # we update the public hand for all players yet to play to the trick
        hand = rolls[player_i].get_player_hand(trick_i).copy()
        hand[s_on_play, card[s_on_play]] -= 1
        if player_i == 1: # dummy
            rolls[2].x_in[s_on_play & (n_cards_seen < 3), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 3)]
            rolls[3].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]
            rolls[0].x_in[s_on_play & (n_cards_seen < 1), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 1)]
        
        if player_i == 3: # declarer on play and dummy hasn't played yet (i.e. n_cards_seen < 2)
            rolls[1].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]

        n_cards_seen[s_on_play] += 1

    print("trick %s is complete" % trick_i)
    # finding the trick winners
    # creating the one-hot-encoded full trick
    full_trick = np.zeros((n_samples, 4 * 32))
    for i in range(4):
        full_trick[s_all, i*32 + trick_cards[:,i]] = 1

    
    winner_trick_6 = get_trick_winner_i(full_trick, strain)
    leader_trick_7 = (leader_trick_6 + winner_trick_6.reshape(-1)) % 4
    print(np.hstack([trick_cards, leader_trick_6.reshape((-1, 1)), leader_trick_7.reshape((-1, 1))]))

    ## preparing the input for trick 7 (next trick)
    # setting all player hands

    for i in range(4):
        hand = rolls[i].get_player_hand(trick_i).copy()
        player_i_card_index = (i - leader_trick_6) % 4
        card = trick_cards[s_all, player_i_card_index]
        hand[s_all, card] -= 1
        rolls[i].set_player_hand(trick_i + 1, hand)


    assert np.sum(rolls[0].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[0].get_player_hand(6)[0]) == 7
    assert np.sum(rolls[1].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[1].get_player_hand(6)[0]) == 7
    assert np.sum(rolls[2].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[2].get_player_hand(6)[0]) == 7
    assert np.sum(rolls[3].get_player_hand(5)[0]) == 8
    assert np.sum(rolls[3].get_player_hand(6)[0]) == 7

    # setting all public hands
    rolls[0].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
    rolls[1].set_public_hand(trick_i+1, rolls[3].get_player_hand(trick_i))
    rolls[2].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
    rolls[3].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))

    # set last trick. and current trick is all zeros. last trick lead index is 0.
    for i in range(4):
        rolls[i].x_in[:,trick_i+1,64:192] = full_trick
        rolls[i].x_in[:,trick_i+1,192:288] = np.zeros((n_samples, 3*32))
        rolls[i].x_in[:,trick_i+1,288:292] = np.zeros((n_samples, 4))
        rolls[i].x_in[s_all,trick_i+1,288 + leader_trick_6] = 1  # last trick lead player index
        rolls[i].x_in[:,trick_i+1,292] = 4  # level
        rolls[i].x_in[:,trick_i+1,293:] = strain

    ###
    ## now we roll the next tricks in a loop

    leader_prev_trick = leader_trick_6
    leader_trick = leader_trick_7

    start_t = time.time()
    prediction_times = []

    for trick_i in (6, 7, 8, 9, 10, 11, 12):
        print('\n\n==== TRICK %s\n\n' % trick_i)

        n_cards_expected = 4 * np.ones(n_samples, dtype=np.int)
        n_cards_seen = np.zeros(n_samples, dtype=np.int)

        # we cycle through each player twice (lefty -> dummy -> righty -> declarer). 
        # we take a prediction, bit if it was not the player's turn, we don't update the game state
        trick_cards = -np.ones((n_samples, 4), dtype=np.int)
        for player_i in [0, 1, 2, 3, 0, 1, 2, 3]:
            if np.all(n_cards_seen >= n_cards_expected):
                break  # we are done with this trick. no need to cycle further

            s_on_play = ((n_cards_seen > 0) | (leader_trick == player_i)) & (n_cards_seen < n_cards_expected)
            # we predict the next card
            p_start_t = time.time()
            p_next_card_softmax = players[player_i].next_cards_softmax(rolls[player_i].x_in[:,:(trick_i + 1),:])
            prediction_times.append(time.time() - p_start_t)
            card_softmax = follow_suit(
                p_next_card_softmax,
                BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_player_hand(),
                BinaryInput(rolls[player_i].x_in[:,trick_i,:]).get_this_trick_lead_suit(),
            )
            card = np.argmax(card_softmax, axis=1)

            trick_cards[s_on_play, n_cards_seen[s_on_play]] = card[s_on_play]
                
            # we update the current trick for all players yet to play to this trick
            rolls[(player_i + 1) % 4].x_in[s_on_play & (n_cards_seen < 3), trick_i, 192 + 2*32 + card[s_on_play & (n_cards_seen < 3)]] = 1
            rolls[(player_i + 2) % 4].x_in[s_on_play & (n_cards_seen < 2), trick_i, 192 + 1*32 + card[s_on_play & (n_cards_seen < 2)]] = 1
            rolls[(player_i + 3) % 4].x_in[s_on_play & (n_cards_seen < 1), trick_i, 192 + 0*32 + card[s_on_play & (n_cards_seen < 1)]] = 1

            # we update the public hand for all players yet to play to the trick
            hand = rolls[player_i].get_player_hand(trick_i).copy()
            hand[s_on_play, card[s_on_play]] -= 1
            if player_i == 1: # dummy
                rolls[2].x_in[s_on_play & (n_cards_seen < 3), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 3)]
                rolls[3].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]
                rolls[0].x_in[s_on_play & (n_cards_seen < 1), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 1)]
            
            if player_i == 3: # declarer on play and dummy hasn't played yet (i.e. n_cards_seen < 2)
                rolls[1].x_in[s_on_play & (n_cards_seen < 2), trick_i, 32:64] = hand[s_on_play & (n_cards_seen < 2)]

            n_cards_seen[s_on_play] += 1

        print("trick %s is complete" % trick_i)
        # finding the trick winners
        # creating the one-hot-encoded full trick
        full_trick = np.zeros((n_samples, 4 * 32))
        for i in range(4):
            full_trick[s_all, i*32 + trick_cards[:,i]] = 1

        
        winner_trick = get_trick_winner_i(full_trick, strain)
        leader_next_trick = (leader_trick + winner_trick.reshape(-1)) % 4
        print(np.hstack([trick_cards, leader_trick.reshape((-1, 1)), leader_next_trick.reshape((-1, 1))]))

        if trick_i < 12:
            ## preparing the input for next trick
            # setting all player hands

            for i in range(4):
                hand = rolls[i].get_player_hand(trick_i).copy()
                player_i_card_index = (i - leader_trick) % 4
                card = trick_cards[s_all, player_i_card_index]
                hand[s_all, card] -= 1
                rolls[i].set_player_hand(trick_i + 1, hand)


            # assert np.sum(rolls[0].get_player_hand(5)[0]) == 8
            # assert np.sum(rolls[0].get_player_hand(6)[0]) == 7
            # assert np.sum(rolls[1].get_player_hand(5)[0]) == 8
            # assert np.sum(rolls[1].get_player_hand(6)[0]) == 7
            # assert np.sum(rolls[2].get_player_hand(5)[0]) == 8
            # assert np.sum(rolls[2].get_player_hand(6)[0]) == 7
            # assert np.sum(rolls[3].get_player_hand(5)[0]) == 8
            # assert np.sum(rolls[3].get_player_hand(6)[0]) == 7

            # setting all public hands
            rolls[0].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
            rolls[1].set_public_hand(trick_i+1, rolls[3].get_player_hand(trick_i))
            rolls[2].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))
            rolls[3].set_public_hand(trick_i+1, rolls[1].get_player_hand(trick_i))

            # set last trick. and current trick is all zeros. last trick lead index is 0.
            for i in range(4):
                rolls[i].x_in[:,trick_i+1,64:192] = full_trick
                rolls[i].x_in[:,trick_i+1,192:288] = np.zeros((n_samples, 3*32))
                rolls[i].x_in[:,trick_i+1,288:292] = np.zeros((n_samples, 4))
                rolls[i].x_in[s_all,trick_i+1,288 + leader_trick] = 1  # last trick lead player index
                rolls[i].x_in[:,trick_i+1,292] = 4  # level
                rolls[i].x_in[:,trick_i+1,293:] = strain

            leader_trick = leader_next_trick

    

    end_t = time.time()
    for i, t in enumerate(prediction_times):
        print(i, t)
    print('total_time', end_t - start_t)
    print('print RNN prediction time', sum(prediction_times))

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
