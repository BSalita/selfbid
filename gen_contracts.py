import sys

from redeal import Deal

levels = [1, 2, 3, 4, 5, 6, 7]
strains = ['C', 'D', 'H', 'S', 'N']
doubles = ['', 'X', 'XX']
declarers = ['N', 'E', 'S', 'W']


def get_contract_scores(deal):
    result = []

    for declarer in declarers:
        for level in levels:
            for strain in strains:
                for double in doubles:
                    contract = '{}{}{}{}'.format(level, strain, double, declarer)
                    nonvuln_score = deal.dd_score(contract, False)
                    vuln_score = deal.dd_score(contract, True)
                    result.append([contract, nonvuln_score, vuln_score])

    return result


def write_deal_contract_score(deal, contract_scores):
    print(deal)
    for row in contract_scores:
        print('\t'.join(map(str, row)))


def generate(n):
    dealer = Deal.prepare({})
    i = 0
    while i < n:
        deal = dealer()
        yield deal
        i += 1


if __name__ == '__main__':
    n = int(sys.argv[1])
    
    for deal in generate(n):
        write_deal_contract_score(deal, get_contract_scores(deal))
