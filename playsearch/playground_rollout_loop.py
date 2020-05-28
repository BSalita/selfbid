import numpy as np
import time

from rollout import Rollout, PlayerRollout
from player import BatchPlayer, BatchPlayerLefty, follow_suit, get_trick_winner_i
from binary import *

from playground_rollout import load_players, init_rollout

# we do the rollout in a loop now.
# we are more advanced now :)

def main():

    players = load_players()

    n_samples = 128
    strain = np.zeros((n_samples, 5))
    strain[:,1] = 1 # we set spades trumps. TODO: how to avoid repeat-hardcoding this here?
    rollout = init_rollout(n_samples, players)

    s_all = np.array([True] * n_samples)  # boolean array used to select all samples

    rolls = rollout.player_rollouts

    start_t = time.time()
    prediction_times = []

    leader_trick_alltricks = np.zeros((13, n_samples), dtype=np.int)
    #leader_trick_alltricks[0, :] = np.ones(n_samples, dtype=np.int)
    n_cards_expected_alltricks = 4 * np.ones((13, n_samples), dtype=np.int)
    #n_cards_expected_alltricks[0, :] = 3 * np.ones(n_samples, dtype=np.int)
    n_cards_seen_alltricks = np.zeros((13, n_samples), dtype=np.int)
    n_cards_seen_alltricks[0, :] = 1

    trick_cards_alltricks = - np.ones((13, n_samples, 4), dtype=np.int)
    winner_trick_alltricks = - np.ones((13, n_samples, 1), dtype=np.int)

    trick_cards_alltricks[0, :, 0] = 17  # setting the opening lead
    rolls[1].x_in[s_all,0,192 + 2*32 + trick_cards_alltricks[0, :, 0]] = 1    # setting the current trick
    rolls[2].x_in[s_all,0,192 + 1*32 + trick_cards_alltricks[0, :, 0]] = 1    # setting the current trick
    rolls[3].x_in[s_all,0,192 + 0*32 + trick_cards_alltricks[0, :, 0]] = 1    # setting the current trick

    total_tricks_won = np.zeros((n_samples, 4), dtype=np.int)

    for trick_i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        print('\n\n==== TRICK %s\n\n' % trick_i)

        n_cards_expected = n_cards_expected_alltricks[trick_i, :]
        n_cards_seen = n_cards_seen_alltricks[trick_i, :]

        leader_trick = leader_trick_alltricks[trick_i, :]

        # we cycle through each player twice (lefty -> dummy -> righty -> declarer). 
        # we take a prediction, bit if it was not the player's turn, we don't update the game state
        trick_cards = trick_cards_alltricks[trick_i, :, :]
        assert trick_cards.shape == (n_samples, 4)
        for player_i in [0, 1, 2, 3, 0, 1, 2, 3]:
            if trick_i == 0 and player_i == 0:
                continue # we never predict the opening lead (that's a different model)
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

        #print("trick %s is complete" % trick_i)
        # finding the trick winners
        # creating the one-hot-encoded full trick
        full_trick = np.zeros((n_samples, 4 * 32))
        for i in range(4):
            full_trick[s_all, i*32 + trick_cards[:,i]] = 1

        
        winner_trick = get_trick_winner_i(full_trick, strain)
        leader_next_trick = (leader_trick + winner_trick.reshape(-1)) % 4
        print(np.hstack([trick_cards, leader_trick.reshape((-1, 1)), leader_next_trick.reshape((-1, 1))]))

        winner_trick_alltricks[trick_i] = winner_trick

        #import pdb; pdb.set_trace()
        total_tricks_won[s_all, leader_next_trick] += 1

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

            leader_trick_alltricks[trick_i + 1] = leader_next_trick

    

    end_t = time.time()
    for i, t in enumerate(prediction_times):
        print(i, t)

    print(total_tricks_won)

    print('total_time', end_t - start_t)
    print('print RNN prediction time', sum(prediction_times))

    #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
