import sys

from kbb.bidder import KnowledgeBasedBidder
from core.hand import Hand
from core.callhistory import Vulnerability, CallHistory
from core.position import Position
from core.call import Pass

kbb_bidder = KnowledgeBasedBidder()

def hands_from_deal_str(deal_str):
    result = []
    for hand in deal_str.split('\t'):
        cdhs_str = '.'.join(map(lambda s: s[1:], reversed(hand.split())))
        result.append(Hand.from_cdhs_string(cdhs_str))
    return result

def get_call_history(hands, board_number):
    call_history = CallHistory([], Position.from_index((board_number + 3) % 4), Vulnerability.from_board_number(board_number))

    while len(call_history.calls) < 4 or not call_history.is_complete():
        whos_turn = call_history.position_to_call().index
        call = kbb_bidder.find_call_for(hands[whos_turn], call_history) or Pass()
        call_history.calls.append(call)
    
    return call_history

if __name__ == '__main__':
    board_number = 0
    for line in sys.stdin:
        if not line.startswith(' S'):
            sys.stdout.write(line)
            continue
        board_number += 1
        hands = hands_from_deal_str(line.strip().replace('  ', '\t'))
        call_history = get_call_history(hands, board_number)
        sys.stdout.write(line)
        sys.stdout.write(' %s %s %s\n' % (['N', 'E', 'S', 'W'][call_history.dealer.index], call_history.vulnerability.name, call_history.calls_string()))
    
