# ================================ Imports ============================================
import numpy as np
from random import choice
import copy
import traceback
from time import time


# ====================== Definition of constants ======================================
PLACES_PER_SEGMENT = 16  # DO NOT CHANGE!!
RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠ ♡ ♠'.split()
DEAL_ORDER = [5, 4, 4]
AVERAGE_GAME_LENGTH = 35
AVERAGE_GAME_LENGTHS = {"2": 45, "4": 150, "6": 350}

# ================================= Classes ===========================================

def print_block(s):
    slen = len(s)
    print("=" * (slen + 4))
    print(f"| {s} |")
    print("=" * (slen + 4))

def print_important(s):
    print(f"[!]: {s}")

START_PAWN = 4
CARD_TO_REPR = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"] * 10
ACTION_TUPLES = [(0, 0) for i in range(20)]
for action_number in range(20):
    cardidx, pawn = ACTION_TUPLES[action_number] = (action_number // 4, action_number % 4)

ACE_START_AN = 49
KING_START_AN = 48
ACE = 12
KING = 11
QUEEN = 10
JACK = 9
CTV = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1] * 7
PAWN_NAMES=["1st","2nd","3rd","4th","start"]
ACTION_IDX_TO_PAWN_VALUE = np.empty(50, dtype=tuple)
ACTION_IDX_TO_NAME = np.empty(50, dtype='object')
CONCISE_ACTION_IDX_TO_NAME = np.empty(50, dtype='object')

ACTION_IDX_TO_CARD_PAWN = np.empty(50, dtype=tuple)
for idx in range(50):
    if idx == ACE_START_AN:
        card = ACE
    elif idx == KING_START_AN:
        card = KING
    else:
        card = idx // 4
        if card == KING:
            card = ACE
    pawn = idx % 4 if idx not in [ACE_START_AN, KING_START_AN] else START_PAWN 
    value = CTV[card]
    cardname = CARD_TO_REPR[card]
    pawnname = str(pawn)
    name = f"{cardname:<5} on {PAWN_NAMES[pawn]:>5}"
    concise_name = f"{cardname} on {PAWN_NAMES[pawn]}"
    ACTION_IDX_TO_NAME[idx] = name
    CONCISE_ACTION_IDX_TO_NAME[idx] = concise_name
    ACTION_IDX_TO_CARD_PAWN[idx] = tuple((card, pawn))
    ACTION_IDX_TO_PAWN_VALUE[idx] = tuple((pawn, value))

# print(f"n: name, (card, pawn), (pawn, value)")
# for i, (name, pv, cp) in enumerate(zip(ACTION_IDX_TO_NAME,ACTION_IDX_TO_PAWN_VALUE,ACTION_IDX_TO_CARD_PAWN)):
#     print(f"{i}: {name}, {cp}, {pv}")

def is_capture(action_number, field, base, start, verbose=False):
    boardlen = len(field[0])
    if action_number in [ACE_START_AN, KING_START_AN]:
        endpos = 0
    else:
        pawn, value = ACTION_IDX_TO_PAWN_VALUE[action_number]
        startpos = np.nonzero(field[pawn])[0]
        assert len(startpos) == 1
        startpos = startpos[0]
        endpos = min(boardlen, startpos + value)
    if endpos == boardlen:
        return None, 0, 0, 0, 0
    for i in range(4, len(field)):
        if field[i][endpos]:
            basesum = base[i]
            startsum = start[i]
            player = i - 3
            pos = (endpos - 16 * player) % boardlen + 1
            possum = basesum * (boardlen + 1)
            pawnidxs = np.nonzero(field[i])[0]
            for idx in pawnidxs:
                possum += (idx - 16 * player) % boardlen + 1
            if verbose:
                print(f"[!]: Move captures opponent {player}")
                print(f"    Position    : {pos:.0f}")
                print(f"    Position sum: {possum:.0f}")
                print(f"    In base     : {basesum:.0f}")
                print(f"    In start    : {startsum:.0f}")
            return player, pos, possum, basesum, startsum
    return None, 0, 0, 0, 0

def action_tuple_to_action_number(action_tuple, player_obs, cards):
    """
    Make sure that the action tuple and player obs are synchronized, i.e.
    they are either both in the obs-shifted version or both not.
    """
    cardidx, pawn = action_tuple
    card = cards[cardidx]
    card = card % 13
    # print(action_tuple)
    # print(player_obs)
    # print_cards(cards)
    if card == ACE and player_obs[pawn] == 0:
        return ACE_START_AN
    if card == ACE and player_obs[pawn] != 0:
        return KING * 4 + pawn
    if card == KING:
        if player_obs[pawn] != 0:
            print(f"action_tuple_to_action_number: Invalid action KING on pawn not in homebase")
            breakpoint()
        return KING_START_AN
    action_number = card * 4 + pawn
    return int(action_number)

def tupled_action_space_to_action_mask(action_space, player_obs, cards):
    # print(f"Player obs {player_obs}")
    sorted_player_obs = sorted(player_obs)
    # print(f"Sorted player obs {sorted_player_obs}")
    action_mask = np.zeros(50, dtype=bool)
    for action_tuple in action_space:
        pawnobs = player_obs[action_tuple[1]]
        for i, ob in enumerate(sorted_player_obs):
            if ob == pawnobs:
                # print(f"Match found at index {i} for ob = {ob}")
                pawn = i
        shifted_action_tuple = (action_tuple[0], pawn)
        # print(f"Shifted tuple from {action_tuple} to {shifted_action_tuple}")
        action_number = action_tuple_to_action_number(shifted_action_tuple, sorted_player_obs, cards)
        action_mask[action_number] = True
    return action_mask

def obs_shift_action_tuple(action_tuple, _player_obs, reverse=False):
    # print(f"Shifting action tuple (reverse is {reverse})!")
    player_obs = sorted(_player_obs)
    # print(f"_player_obs: {_player_obs}")
    # print(f"player_obs : {player_obs}")
    pawn = action_tuple[1]
    # print(f"pawn       : {pawn}")
    if not reverse:
        pawnobs = _player_obs[pawn]
        # print(f"pawnobs    : {pawnobs}")
        for i, ob in enumerate(player_obs):
            if pawnobs == ob:
                spawn = i
    else:
        pawnobs = player_obs[pawn]
        # print(f"pawnobs    : {pawnobs}")
        for i, ob in enumerate(_player_obs):
            if pawnobs == ob:
                spawn = i
    # print(f"spawn      : {spawn}")

    return (action_tuple[0], spawn)

def action_number_to_action_tuples(action_number, player_obs, cards, *, nocard=None):
    # print(f"Getting action tuples for number {action_number}")
    # print(player_obs)
    # print_cards(cards)
    if action_number in [ACE_START_AN, KING_START_AN]:
        # print(f"Is start type")
        action_card = ACE if action_number == ACE_START_AN else KING
        cardidx = []
        for i, card in enumerate(cards):
            if card is not nocard:
                if action_card == card % 13:
                    cardidx.append(i)
        pawns = []
        for pawn in range(4):
            if player_obs[pawn] == 0:
                pawns.append(pawn)
        action_tuples = [tuple((i, j)) for i in cardidx for j in pawns]
        # print(f"Returning {action_tuples}")
    # print(f"Is normal type")
    else:
        action_card, action_pawn = ACTION_IDX_TO_CARD_PAWN[action_number]
        cardidx = []
        for i, card in enumerate(cards):
            if card is not nocard:
                if action_card == card % 13:
                    cardidx.append(i)
        action_tuples =  [tuple((i, action_pawn)) for i in cardidx]
        # print(f"Returning {action_tuples}")
    if not action_tuples:
        print(f"Action tuples are empty, bad action number {action_number}? player_ob = {player_obs}, cards")
        print_cards(cards, nocard=nocard)
        breakpoint()
    return action_tuples

def print_cards(cards, nocard=None):
    s = "["
    for i, card in enumerate(cards):
        if card is nocard:
            s += "none"
        else:
            s += CARD_TO_REPR[card]
        if i != len(cards) - 1:
            s += ", "
    s += "]"
    print(s)
    
def return_search_results(return_type, action_mask, tockgame_player_obs, tockgame_info, results, *, verbose=False, check_normalization=False, tau=None, action_values=None):
    cards = tockgame_info['cards']
    tockgame_space = tockgame_info['space']
    if return_type == "unique_probabilities":
        action_numbers = np.arange(50)[action_mask]
        action_prob = results / (np.sum(results) + 1e-10)
        if len(np.nonzero(action_prob)[0]) == 0:
            action_prob = 1/len(action_prob) * np.ones(len(action_prob))
            # print(f"Fixed probabilities to {action_prob}!")
        # print(f"Returning {action_prob}, {action_numbers}")
        return action_prob, action_numbers
    else:
        action_tuples, result, best_action = action_mask_to_tuple_action_space(action_mask, tockgame_player_obs, cards, action_probabilities=results, verbose=verbose, check_normalization=check_normalization, tau=tau, action_values=action_values)
        for i in range(len(action_tuples)):
            if action_tuples[i] != tockgame_space[i]:
                print(f"return_search_results: action tuple mismatch")
                print(f"Got action tuples {action_tuples}, which is not equal to game action space {tockgame_space}")
                print(f"Error occurred when matching tuple {action_tuples[i]} to {tockgame_space[i]}")
                breakpoint()
        if return_type == "best_action":
            return best_action
        else:
            return result

def action_mask_to_tuple_action_space(action_mask, _player_obs, cards, *, action_probabilities=None, verbose=False, check_normalization=True, tau=None, action_values=None):
    """
    Returns ordered action tuples if action probabilities not given, else a tuple ordered action tuples, action probabilities, best_action
    Reorders obs for calculation and returns pawn-rotated probabilities to match TockGame! This assumes that probabilities are in the FastTockGame order!!
    """
    player_obs = sorted(_player_obs)
    # print(f"Unsorted obs: {_player_obs}")
    # print(f"Sorted obs  : {player_obs}")
    obs_order = [key for key, value in sorted(enumerate(_player_obs), key=lambda x: x[1])]
    # print(f"Obs order   : {obs_order}")

    if verbose:
        print(f"============================================")
        print(f"Converting action mask to tuple action space")
        print(f"   Current player's pawn locations are:\n\t{player_obs}")
        print(f"   Current player's cards are:\n\t", end='')
        print_cards(cards)
        print_action_space(action_mask, action_probabilities=action_probabilities, check_normalization=check_normalization, action_values=action_values)
    action_indices = np.arange(50)[action_mask]
    # print_action_space(action_mask)
    action_tuples = []
    if action_probabilities is not None:
        if len(action_probabilities) == 50:
            valid_action_prob = action_probabilities[action_mask]
        else:
            valid_action_prob = action_probabilities
            assert len(valid_action_prob) == len(action_indices), f"action_mask_to_tuple_action_space: incorrect probabilities supplied, lengths {len(valid_action_prob)} and {len(action_indices)} do not match"
        if check_normalization:
            if abs(sum(valid_action_prob) - 1) > 1e-3:
                print(f"action_mask_to_tuple_action_space: valid_action_prob is {valid_action_prob} with sum {sum(valid_action_prob)} != 1")
                breakpoint()
            if tau is not None:
                scaled_valid_action_prob = valid_action_prob ** ( 1/tau ) 
                valid_action_prob = scaled_valid_action_prob / np.sum(scaled_valid_action_prob)
                print(f"Rescaled values with tau of {tau}")
        best_action_idx = action_indices[np.argmax(valid_action_prob)]
    action_prob = []
    for i, idx in enumerate(action_indices):
        idx_tuple_list = action_number_to_action_tuples(idx, player_obs, cards)
        obs_converted_itl = [(idx_tuple_list[i][0], obs_order[idx_tuple_list[i][1]]) for i in range(len(idx_tuple_list))]
        # print(f"idx_tuple_list              : {idx_tuple_list} -> {obs_converted_itl} (obs-converted)")
        action_tuples += obs_converted_itl
        if action_probabilities is not None and idx == best_action_idx:
            best_action = choice(obs_converted_itl)
        ntuples = len(idx_tuple_list)
        if action_probabilities is not None:
            if check_normalization:
                action_prob += [valid_action_prob[i] / ntuples] * ntuples
            else:
                action_prob += [valid_action_prob[i]] * ntuples

    if action_probabilities is None:
        action_tuples = sorted(action_tuples)
        if verbose:
            print(f"Got action tuples:")
            for action_tuple in action_tuples:
                print(f"  {action_tuple_to_string(action_tuple, cards)}")
            print(f"============================================")
        
        return action_tuples
    action_tuples_order = [[i, value] for i, value in sorted(enumerate(action_tuples), key=lambda x: x[1])]
    argorder = [action_tuples_order[i][0] for i in range(len(action_tuples))]
    # print(argorder)
    # breakpoint()
    action_prob = np.array(action_prob)[argorder]
    ordered_action_tuples = [action_tuples_order[i][1] for i in range(len(action_tuples))]
    dirichlet_noise = np.random.dirichlet([.2] * len(action_prob))
    action_prob = .999 * action_prob + .001 * dirichlet_noise
    if verbose:
        print(f"Got action tuples:")
        for action_tuple, p in zip(ordered_action_tuples, action_prob):
            if check_normalization:
                p_str = f"{p*100:>4.1f}%"
            else:
                p_str = f"{p:>6.3f}"
            if action_tuple != best_action:
                print(f"  {action_tuple_to_string(action_tuple, cards):<10} -> {p_str}")
            else:
                print(f"\033[1m  {action_tuple_to_string(action_tuple, cards):<10} -> {p_str}  (chosen as best)\033[0m")

        print(f"============================================")
    
    return ordered_action_tuples, action_prob, best_action

def action_tuple_to_string(action_tuple, player_cards):
    card_idx, pawn = action_tuple
    card_name = CARD_TO_REPR[player_cards[card_idx]]
    card_str = f"{card_name:<5}- c{card_idx}"
    return f"({card_str:<8}, {PAWN_NAMES[pawn]:>4})"

def print_action_space(action_space, *, action_probabilities=None, sort=True, check_normalization=True, print_delimiters=True, capture_info=None, action_values=None):
    if print_delimiters:
        print(f"---------------------------------------------")
    print(f"Printing action space{' with probabilities' if action_probabilities is not None and check_normalization else ''}")
    valid_action_names = ACTION_IDX_TO_NAME[action_space]
    action_indices = np.arange(50)[action_space]
    if action_probabilities is not None:
        if len(action_probabilities) == 50:
            valid_action_prob = action_probabilities[action_space]
        else:
            valid_action_prob = action_probabilities
            assert len(valid_action_prob) == len(action_indices), f"print_action_space: incorrect probabilities supplied, lengths {len(valid_action_prob)} and {len(action_indices)} do not match"
        if check_normalization:
            if abs(sum(valid_action_prob) - 1) > 1e-3:
                print(f"print_action_space: valid_action_prob is {valid_action_prob} with sum {sum(valid_action_prob)} != 1")
                breakpoint()

        if sort:
            order = np.argsort(valid_action_prob)[::-1]
            valid_action_prob = valid_action_prob[order]
            valid_action_names = valid_action_names[order]
            action_indices = action_indices[order]
            if action_values is not None:
                action_values = action_values[order]
    for i, (action_name, action_idx) in enumerate(zip(valid_action_names, action_indices)): 
        p_str = ""
        if action_probabilities is not None:
            if check_normalization:
                p_str = f" -> {valid_action_prob[i]*100:>5.1f}%"
            else:
                p_str = f" -> {valid_action_prob[i]:>6.3f}"
        v_str = ""
        if action_values is not None:
            v_str = f"  (v = {action_values[i]:>5.3f})"
        print(f"\t{action_name}{p_str}{v_str}\t(n{action_idx})")
        if capture_info:
            is_capture(action_idx, *capture_info, verbose=True)
    if print_delimiters:
        print(f"---------------------------------------------")

class StateWatchdog(list):
    def __setitem__(self, key, value):
        print(f"DEBUG: state[{key}] changed to {value}")
        traceback.print_stack()
        super().__setitem__(key, value)


def set_globals(nplayers=2):
    global ALL_BITMASKS, PLAYER_BITMASKS, POS_MASKS, BOARD_LEN, NPLAYERS,\
            CARD_TO_VALUE, KING_VALUE, ACE_VALUE, PLAYER, ROUND, OFFSETS, CURRENT_PLAYER,\
            ABSLOCS, PLAYER_ABSLOCS, RELLOCS, PLAYER_CARDS, CARDS, CARD_PTR,\
            ACTION_SPACE, GAMESPACE_LEN, DECK_LEN, PL_ABSL_PTR,\
            RELLOCS_PTR, PLAYER_CARDS_PTR, PREVIOUS_FOLD, NEW_GAME, CURRENT_ACTION,\
            MOVE_NUMBER, RELLOC_TABLE, ABSLOC_TABLE, ABSLOC_VALUE_TABLE, RELLOC_START_PTR,\
            ACTION_TUPLES, MAX_HIS_LEN, PLAYER_ALL_CARDS, NOCARD, ALL_RELLOCS, PREVIOUS_PLAY, CHOSEN_ACTION
    NPLAYERS = nplayers
    BOARD_LEN = NPLAYERS * 16
    DECK_LEN = NPLAYERS * 13
    NOCARD = DECK_LEN
    MAX_HIS_LEN = NPLAYERS * 100
    OFFSETS = [player * 16 for player in range(NPLAYERS)]
    CARD_TO_VALUE = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1] * NPLAYERS
    PREVIOUS_FOLD = [-3, -3]
    NEW_GAME = [-1, -1]
    PREVIOUS_PLAY = [-2, -2]

    KING_VALUE = 0
    ACE_VALUE = 1

    # set indices for gamestate variables in FastTockGame environment
    CURRENT_PLAYER = 0
    ROUND = 1
    ABSLOCS = 2

    PL_ABSL_PTR = 3
    PLAYER_ABSLOCS = slice(3, 3 + NPLAYERS)

    RL0 = slice(3 + NPLAYERS, 7 + NPLAYERS)
    RL1 = slice(7 + NPLAYERS, 11 + NPLAYERS)
    RL2 = slice(11 + NPLAYERS, 15 + NPLAYERS)
    RL3 = slice(15 + NPLAYERS, 19 + NPLAYERS)
    RL4 = slice(19 + NPLAYERS, 23 + NPLAYERS)
    RL5 = slice(23 + NPLAYERS, 27 + NPLAYERS)
    RELLOCS = (RL0, RL1, RL2, RL3, RL4, RL5)
    ALL_RELLOCS = slice(3 + NPLAYERS, 3 +  5*NPLAYERS)

    RL_P0 = 3 + NPLAYERS
    RL_P1 = 7 + NPLAYERS
    RL_P2 = 11 + NPLAYERS
    RL_P3 = 15 + NPLAYERS
    RL_P4 = 19 + NPLAYERS
    RL_P5 = 23 + NPLAYERS
    RELLOCS_PTR = (RL_P0, RL_P1, RL_P2, RL_P3, RL_P4, RL_P5)
    RELLOC_START_PTR = 3 + NPLAYERS

    C0 = slice(3+5*NPLAYERS, 8 + 5*NPLAYERS)
    C1 = slice(8+5*NPLAYERS, 13 + 5*NPLAYERS)
    C2 = slice(13+5*NPLAYERS, 18 + 5*NPLAYERS)
    C3 = slice(18+5*NPLAYERS, 23 + 5*NPLAYERS)
    C4 = slice(23+5*NPLAYERS, 28 + 5*NPLAYERS)
    C5 = slice(28+5*NPLAYERS, 33 + 5*NPLAYERS)
    PLAYER_CARDS = (C0, C1, C2, C3, C4, C5)
    PLAYER_CARDS_PTR = 3 + 5*NPLAYERS
    PLAYER_ALL_CARDS = slice(3 + 5*NPLAYERS, 23 + 5*NPLAYERS)

    CARDS = slice(3 + 10*NPLAYERS, 3 + 23*NPLAYERS)
    CARD_PTR = 4 + 23*NPLAYERS
    ACTION_SPACE = 5 + 23*NPLAYERS
    MOVE_NUMBER = 6 + 23*NPLAYERS
    CURRENT_ACTION = slice(7 + 23*NPLAYERS, 9 + 23*NPLAYERS)

    GAMESPACE_LEN = 9 + 23*NPLAYERS



    # computing bitmasks for blocking rule
    ALL_BITMASKS = [[0 for i in range(13)] for j in range(BOARD_LEN)]
    PLAYER_BITMASKS = [[[0 for i in range(13)] for j in range(BOARD_LEN)] for k in range(NPLAYERS)]
    POS_MASKS = [[0 for i in range(13)] for j in range(BOARD_LEN)]
    for start in range(BOARD_LEN):
        for value in range(13):
            if value == 0:
                continue
            stop = (start + (value if value != 4 else -value)) % BOARD_LEN
            if value == 4:
                _start = (stop - 1) % BOARD_LEN
                stop = (start - 1) % BOARD_LEN
            else:
                _start = start
            if stop > _start:
                mask = ~(~0 << (stop - _start)) << (_start + 1)
            else:
                mask = ~ (~(~0 << (_start - stop)) << (stop + 1))

            POS_MASKS[start][value] = mask
    all_start_mask = 0
    for player in range(NPLAYERS):
        all_start_mask <<= 16
        all_start_mask += 1
    for start in range(BOARD_LEN):
        for value in range(13):
            ALL_BITMASKS[start][value] = POS_MASKS[start][value] & all_start_mask
    for player in range(NPLAYERS):
        for start in range(BOARD_LEN):
            for value in range(13):
                PLAYER_BITMASKS[player][start][value] = POS_MASKS[start][value] & (1 << (player * 16))

    # computing absolute/relative location conversion tables
    RELLOC_TABLE = [[0 for i in range(BOARD_LEN)] for j in range(NPLAYERS)]
    for loc in range(BOARD_LEN):
        for player in range(NPLAYERS):
            RELLOC_TABLE[player][loc] = (loc - OFFSETS[player]) % BOARD_LEN + 1

    ABSLOC_TABLE = [[0 for i in range(BOARD_LEN + 2)] for j in range(NPLAYERS)]
    for loc in range(BOARD_LEN + 1):
        for player in range(NPLAYERS):
            if loc == BOARD_LEN + 1:
                ABSLOC_TABLE[player][loc] = BOARD_LEN  # will throw IndexError later
            else:
                ABSLOC_TABLE[player][loc] = (loc + OFFSETS[player] - 1) % BOARD_LEN

    ABSLOC_VALUE_TABLE = [[0 for i in range(BOARD_LEN)] for j in range(13)]
    for loc in range(BOARD_LEN):
        for value in range(13):
            if value == 0:
                ABSLOC_VALUE_TABLE[value][loc] = BOARD_LEN # will throw IndexError later
            else:
                if value != 4:
                    ABSLOC_VALUE_TABLE[value][loc]= (loc + value) % BOARD_LEN
                else:
                    ABSLOC_VALUE_TABLE[value][loc]= (loc - value) % BOARD_LEN
    
    #Pre-computing all action tuples

def print_masks(start=None, value=None):
    if start == None:
        len_range = range(32)
    else:
        len_range = range(start, start + 1)
    if value == None:
        val_range = range(13)
    else:
        val_range = range(value, value + 1)
    for start in len_range:
        for value in val_range:
            print(f"------------------------")
            print(f"Start {start}, Value {value}")
            print(f"Position mask: ")
            print(print_int(POS_MASKS[start][value], NPLAYERS))
            print(f"All player mask: ")
            print(print_int(ALL_BITMASKS[start][value], NPLAYERS))
            print(f"Per-player mask:")
            for player in range(NPLAYERS):
                print(f"Player {player}: ")
                print(print_int(PLAYER_BITMASKS[player][start][value], NPLAYERS))
            print(f"------------------------")

def print_int(uint, nplayers):
    maxlen = 16 * nplayers
    s = ""
    for i in range(maxlen):
        s += f"{i}{"" if i>9 else " "} {"|| " if i % 8 ==7 else ""}"
    s += "\n"
    for i in range(maxlen):
        s += f"{(uint >> i) & 1}  {"|| " if i % 8 ==7 else ""}"
    s += f" (={uint})\n"
    return s




class FastTockGame:
    def __init__(self, nplayers=2, max_his_len=None, store_rng_his=False, debug=False):
        set_globals(nplayers=nplayers)
        self.debug=debug
        self.rng = np.random.default_rng()
        # self.state = StateWatchdog([0] * GAMESPACE_LEN)
        self.store_rng_his = store_rng_his
        self.reset(nplayers=nplayers, max_his_len=max_his_len)
        self.action_mask = np.zeros(64, dtype=bool)
        self.action_mask_his = []
    
    def get_hash(self):
        return (tuple(self.state[PLAYER_ABSLOCS]), tuple(self.state[PLAYER_ALL_CARDS]))

    def get_NN_hash(self):
        state = self.state
        current_player = state[CURRENT_PLAYER]
        cards =  self.get_cards_list(current_player)
        cards.sort()
        return (tuple(state[ALL_RELLOCS]), tuple(cards), current_player)

    def _check_bits(self):
        try:
            state = self.state
            abslocs = state[ABSLOCS]
            players_abs_loc = state[PLAYER_ABSLOCS]
            rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
            print("Checking bits")
            all_locs = []
            player_locs = [[] for i in range(NPLAYERS)]
            for i in range(BOARD_LEN):
                if (abslocs >> i) & 1:
                    all_locs.append(i)
                for player in range(NPLAYERS):
                    if (players_abs_loc[player] >> i) & 1:
                        rel_loc = (i - 16 * player) % BOARD_LEN + 1
                        player_locs[player].append(rel_loc)
            incorrect = False
            for player in range(NPLAYERS):
                for loc in player_locs[player]:
                    if loc not in rellocs[player]:
                        incorrect = True
                for i in range(4):
                    loc = rellocs[player][i]
                    if loc not in [0, BOARD_LEN + 1]:
                        if loc not in player_locs[player]:
                            incorrect = True
            if incorrect:
                print("Error: location mismatch")
            all_locs = 0
            for i in range(NPLAYERS):
                all_locs |= players_abs_loc[i]
            sum_error = False
            for i in range(BOARD_LEN):
                if ((abslocs >> i) & 1) != ((all_locs >> i) & 1):
                    print(f"Detected sum error at absolute field {i}")
                    sum_error = True
                    break
            if sum_error:
                print("Error: sum mismatch")
                incorrect = True
            print(self._print_abslocs())
            print("Board position:")
            print(self)
            if incorrect:
                breakpoint()
            else:
                print("Bits in order")
        except Exception as e:
            traceback.print_exc()
            breakpoint()

    def _print_abslocs(self):
        abslocs = self.state[ABSLOCS]
        player_abslocs = self.state[PLAYER_ABSLOCS]
        s = ""
        s += "All players:\n"
        s += print_int(abslocs, NPLAYERS)
        for player in range(NPLAYERS):
            s += f"Player {player}:\n"
            s += print_int(player_abslocs[player], NPLAYERS)
        return s


    def _execute_move(self, pawn, value):
        if value == 0:
            value = 1
        nply_range = range(NPLAYERS)
        boardlen = BOARD_LEN
        state = self.state
        abslocs = state[ABSLOCS]
        pl_absl_ptr = PL_ABSL_PTR
        players_abs_loc = state[PLAYER_ABSLOCS]
        current_player = state[CURRENT_PLAYER]
        relloc_current_player = state[RELLOCS[current_player]]
        loc = relloc_current_player[pawn]

        absloc_table = ABSLOC_TABLE
        relloc_table = RELLOC_TABLE
        absloc_value_table = ABSLOC_VALUE_TABLE

        start = absloc_table[current_player][loc]
        stop = absloc_value_table[abs(value)][start]
        
        if loc + value > boardlen:
            state[ABSLOCS] &= ~(1 << start)
            state[pl_absl_ptr + current_player] &= ~(1 << start)
            state[RELLOCS_PTR[current_player] + pawn] = boardlen + 1
            # self._check_bits()
            # print(f"Executed move (pawn: {pawn}, value: {value}) just now")
            # breakpoint()
            return

        # check if another pawn is at the stop position
        if abslocs & (1 << stop):
            rellocs = [state[RELLOCS[player]] for player in nply_range]
            for _player in nply_range:
                if players_abs_loc[_player] & (1 << stop):
                    rel_index_for_player = RELLOC_TABLE[_player][stop]
                    idx = rellocs[_player].index(rel_index_for_player)
                    state[RELLOCS_PTR[_player] + idx] = 0
                    state[pl_absl_ptr + _player] &= ~( 1 << stop)
        state[RELLOCS_PTR[current_player] + pawn] = RELLOC_TABLE[current_player][stop]
        startmask = ~(1 << start)
        stopmask = ( 1 << stop)
        if loc != 0:
            state[ABSLOCS] &= startmask
        state[ABSLOCS] |= stopmask
        if loc != 0:
            state[pl_absl_ptr + current_player] &= startmask
        state[pl_absl_ptr + current_player] |= stopmask
        # self._check_bits()
        # print(f"Executed move (pawn: {pawn}, value: {value}) just now")
        # breakpoint()
        return


    def _validate_move(self, pawn, value):
        # check if pawn makes it to homebase
        # pawn can not be blocked by other player because value < LENGTH_SEGMENT
        if value == 0:
            value = 1
        state = self.state
        abslocs = state[ABSLOCS]
        players_abs_loc = state[PLAYER_ABSLOCS]
        current_player = state[CURRENT_PLAYER]
        loc = state[RELLOC_START_PTR + 4*current_player + pawn]
        all_bitmasks = ALL_BITMASKS
        player_bitmasks = PLAYER_BITMASKS

        start = ABSLOC_TABLE[current_player][loc]
        stop = ABSLOC_VALUE_TABLE[abs(value)][start]
        
        if loc + value > BOARD_LEN:
            return True 
        # check interference with other players pawns

        # first compute absolute positions before and after move
        
        # check if move is blocked by other players pawn in startpostion
        if all_bitmasks[start][abs(value)] & abslocs:
            for player in range(NPLAYERS):
                if player == current_player:
                    continue
                if players_abs_loc[player] & player_bitmasks[player][start][abs(value)]:
                    return False
        return True

    def _update_action_space(self):
        """computes action space"""
        state = self.state
        player = state[CURRENT_PLAYER]
        cards = state[PLAYER_CARDS[player]]
        state[ACTION_SPACE] = 0
        rellocs_player = state[RELLOCS[player]]
        card_values = [CARD_TO_VALUE[c] if (c != NOCARD) else None for c in cards] 
        for action_number, (cardidx, pawn) in enumerate(ACTION_TUPLES):
            card = cards[cardidx]
            if card == NOCARD:
                continue
            valid = False
            loc = rellocs_player[pawn]
            card_value = card_values[cardidx]

            # check if pawn is in startgrid
            if loc == 0:
                # pawn in start grid
                if card_value in [KING_VALUE, ACE_VALUE] :  # heer of aas
                    # place pawn on the field if in startgrid
                    valid =  self._validate_move(pawn, 1)
                else:  
                    # not a valid move
                    valid = False
                
            # check if pawn is in homebase
            elif loc > BOARD_LEN:
                valid = False
            else:
                # pawn on the field
                if card_value == KING_VALUE:
                    valid =  False
                else:
                    valid = self._validate_move(pawn, card_value)
            if valid:
                valid_mask = 1 << action_number
                state[ACTION_SPACE] |= valid_mask
            # print(f"Validated action ({CARD_TO_REPR[card % 13]}, {pawn}) as {'Valid' if valid else 'invalid'}")
        # print(f"Evaluated all actions, action space is now:")
        # print(print_int(state[ACTION_SPACE], NPLAYERS))

    def _get_next_player(self):
        """returns the next player in the playerslist that has at least one card"""
        state = self.state
        oldplayer = state[CURRENT_PLAYER]
        player = oldplayer
        while True:
            player = state[CURRENT_PLAYER] = (player + 1) % NPLAYERS
            self._update_action_space()
            if state[ACTION_SPACE] == 0:
                # print(f"Action space is empty, skippping player {player}")
                # print(self)
                # if not self.resetting_now:
                #     self.history[-1]['cards'].append((self.players.index(player), 1000, player.cards))
                state[PLAYER_CARDS[player]] = [NOCARD] * 5  # remove cards if no actions are possible for player
                state[CURRENT_ACTION] = PREVIOUS_FOLD  # this means the current action is none
                if player == oldplayer:  # handle case if all players lost their cards
                    # print(f"Next round")
                    state[ROUND] += 1
                    self.deal()
            else:
                break  # a next player is found

    def get_action_number(self, action_tuple):
        return action_tuple[0] * 4 + action_tuple[1]

    def step(self, action, gettuple_as=False, action_number_input=True):
        """
        performs action on the current player and returns the gamestate after the action
        """
        state = self.state
        if self.debug:
            print(f"FastTockGame.step: Executing action {action_tuple_to_string(action, self.get_cards_list(state[CURRENT_PLAYER])) if type(action) is tuple else ACTION_IDX_TO_NAME[action]}")
            print(f"Current game state is:")
            print(self)
        try:
            if not action_number_input:
                converted_action = self.convert_action(action)
                
                if not (state[ACTION_SPACE] >> self.get_action_number(converted_action)) & 1:
                    traceback.print_exc()
                    print("Error: action not in action space")
                    breakpoint()
            elif self.action_mask[action] == False:
                    traceback.print_exc()
                    print("Error: action not in action space")
                    breakpoint()
        except IndexError:
            traceback.print_exc()
            breakpoint()

        current_move = state[MOVE_NUMBER]
        history = self.history
        if current_move >= len(history):
            print(
                    f"Error: cannot store more than {len(self.history)} moves,"
                    f"reset with appropriate 'max_his_len' to increase buffer size"
            )
            breakpoint()
        history[current_move] = self.state[:]
        if self.debug:
            print(f"Set history for move {current_move}")
        self.action_mask_his.append(self.action_mask[:])
        current_player = state[CURRENT_PLAYER]
        if self.store_rng_his:
            self.rng_his.append(self.rng.bit_generator.state)
        if action_number_input:
            player_obs = state[RELLOCS[current_player]]
            cards = state[PLAYER_CARDS[current_player]]
            action_tuples = action_number_to_action_tuples(action, player_obs, cards, nocard=NOCARD)
            # print(player_obs)
            # print_cards(cards, nocard=NOCARD)
            # print(action_tuples)
            # breakpoint()
            # choose to move the last pawn if multiple pawns can be moved
            converted_action = action_tuples[-1]
            # print(f"Chose action {converted_action}")
            state[CURRENT_ACTION] = self.convert_action(converted_action, reverse=True)

            # print(f"Set current action to {state[CURRENT_ACTION]}")
            # breakpoint()
        else:
            state[CURRENT_ACTION] = action
        cardidx, pawn = converted_action
        card = state[PLAYER_CARDS[current_player]][cardidx]
        value = CARD_TO_VALUE[card]

        state[MOVE_NUMBER] += 1
        self._execute_move(pawn, value)
        for player in range(NPLAYERS):
            player_rellocs = self.state[RELLOCS[player]]
            for i in range(1, 4):
                if player_rellocs[i] < player_rellocs[i-1]:
                    # if player != current_player:
                        # print(f"Reordering rellocs for player {player} (current was {current_player})")
                    order = np.argsort(player_rellocs)
                    ordered_player_rellocs = np.array(player_rellocs)[order]
                    state[RELLOCS[player]] = ordered_player_rellocs
                    break
        # print(f"Player {state[CURRENT_PLAYER]} executed move {(CARD_TO_REPR[card % 13], pawn)} successfully")
        state[PLAYER_CARDS_PTR + 5*current_player + cardidx] = NOCARD

        # determine the next player 
        self._get_next_player() 
        if self.debug:
            print(f"Executed action successfully. Now game state is:")
            print(self)
        return self._get_gamestate(gettuple_as=gettuple_as)


    def _done(self):
        state = self.state
        rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        """determines if the game has finished and if so, which player won"""

        done, winner = False, -1
        for player in range(NPLAYERS):
            is_winner = True
            for loc in range(4):
                if rellocs[player][loc] != BOARD_LEN + 1:
                    is_winner = False
                    break
            if is_winner:
                done, winner = True, player
                break
        return done, winner

    def get_cards_list(self, player):
        state = self.state
        player_cards = state[PLAYER_CARDS[player]]
        card_list = []
        for card in player_cards:
            if card != NOCARD:
                # print(f"Card is {card} (!= nocard {NOCARD})")
                card_list.append(card)
            # print(f"Card is nocard {NOCARD}")
        return card_list
    
    def convert_action(self, action, reverse=False):
        card, pawn = action
        cards = self.state[PLAYER_CARDS[self.state[CURRENT_PLAYER]]]
        if not reverse:
            for i in range(5):
                if cards[i] == NOCARD:
                    card += 1
                if i == card:
                    return np.array([i, pawn])
        else:
            nnocard = 0
            for i in range(5):
                if cards[i] == NOCARD:
                    nnocard += 1
                elif i == card:
                    return np.array([card - nnocard, pawn])

        print(f"Invalid action card index for action {action} with cards {cards}")
        breakpoint()

    def get_action_space(self, gettuple_as=False):
        state = self.state
        action_space = state[ACTION_SPACE]
        non_converted = [ACTION_TUPLES[i] for i in range(20) if ((action_space >> i) & 1)]
        cards = state[PLAYER_CARDS[state[CURRENT_PLAYER]]]
        converted_cards = self._get_converted_cards(cards)
        converted_tuples = [(converted_cards[non_converted[i][0]], non_converted[i][1]) for i in range(len(non_converted))]
        self.action_mask = np.zeros(50, dtype=bool)
        player_obs = state[RELLOCS[state[CURRENT_PLAYER]]]
        for at in non_converted:
            self.action_mask[action_tuple_to_action_number(at, player_obs, cards)] = True
        if gettuple_as:
            return converted_tuples
        else:
            return self.action_mask.copy()

    def _get_converted_cards(self, cards):
        nocard_count = 0
        converted_cards = list(range(5))
        for i in range(5):
            if cards[i] == NOCARD:
                nocard_count += 1
            converted_cards[i] = i - nocard_count
        return converted_cards
    
    def _get_gamestate(self, gettuple_as=False):
        state = self.state
        current_player = state[CURRENT_PLAYER]
        curr_act  = state[CURRENT_ACTION]
        if curr_act == NEW_GAME:
            curr_act = False
        elif curr_act == PREVIOUS_FOLD:
            curr_act = None
        else:
            curr_act = tuple(curr_act)
        rew = None
        done, winner = self._done()
        obs = rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        info = {
                'player': current_player,
                'cards' : self.get_cards_list(current_player),
                'space' : self.get_action_space(gettuple_as=gettuple_as),
                'action' : curr_act,
                'round' : state[ROUND] + 1,
                'winner' : winner
                }
        return obs, rew, done, info

            # if not self.resetting_now:
            #     self.history[-1]['cards'].append((player, None, DEAL_ORDER[self.round%3]))
    def unmove(self):
        old_move_number = self.state[MOVE_NUMBER] - 1
        if old_move_number >= len(self.history):
            print(
                    f"FastTockGame.unmove: cannot undo more than {len(self.history)} moves,"
                    f"reset with appropriate 'max_his_len' to increase buffer size"
            )
            breakpoint()
        elif old_move_number < 0:
            print(
                    f"FastTockGame.unmove: cannot undo any moves; no history yet."
            )
            breakpoint()
        if self.debug:
            print_important(f"FastTockGame.unmove: unmoving from move {self.state[MOVE_NUMBER]} to move {old_move_number}")

        if self.store_rng_his:
             self.rng.bit_generator.state = self.rng_his.pop()
        self.state = self.history[old_move_number][:]
        self.action_mask = self.action_mask_his.pop()

    def set_state(self, state):
        self.state = state[:]

    def get_state(self):
        return self.state[:]
            
    def reset(self, nplayers=None, max_his_len=None):
        """resets the game to the starting position and returns the gamestate"""
        if not nplayers:
            nplayers = NPLAYERS
        if not max_his_len:
            max_his_len = MAX_HIS_LEN
        if self.store_rng_his:
            self.rng_his = []
        self.action_mask_his = []
        # reset all players
        self.state = [0] * (GAMESPACE_LEN)
        # self.state = StateWatchdog([0] * GAMESPACE_LEN)
        self.history = [[0]*GAMESPACE_LEN for i in range(max_his_len)]
        if self.debug:
            print("---------------------")
            print_important("Resetting game state!")
            print(f"State has len {len(self.state)}")
        state = self.state
        state[CURRENT_PLAYER] = NPLAYERS - 1
        self.resetting_now = True
        # state[ABSLOCS] = 0
        # state[PLAYER_ABSLOCS] = [0] * NPLAYERS
        for i in range(NPLAYERS):
            # state[RELLOCS[i]] = [0] * 4
            state[PLAYER_CARDS[i]] = [NOCARD] * 5
        # state[ROUND] = 0
        state[CARDS] = range(13*NPLAYERS)
        # state[CARD_PTR] = 0
        state[CURRENT_ACTION] = NEW_GAME
        # state[MOVE_NUMBER] = 0
        self._shuffle()
        self.deal()
        self._get_next_player()
        self.resetting_now = False
        if self.debug:
            print_important("Reset game state successfully")
            print(f"State has len {len(self.state)}")
            print(self)
            print("---------------------")
        return self._get_gamestate()


    def _shuffle(self):
        state = self.state
        # self.card_history.append(state[CARDS][:])
        cards = state[CARDS]
        self.rng.shuffle(cards)
        state[CARDS] = cards
        state[CARD_PTR] = 0

    def deal(self, ncards=None):
        state = self.state
        card_ptr = state[CARD_PTR]
        cards = state[CARDS]
        if ncards is None:
            ncards = DEAL_ORDER[state[ROUND] % 3]
        for player in range(NPLAYERS):
            hand = [NOCARD] * 5
            try:
                for _ in range(ncards):
                    # If we reached the end of the cards, reshuffle
                    if card_ptr >= DECK_LEN:
                        self._shuffle()
                        cards = state[CARDS]
                        card_ptr = state[CARD_PTR]
                    hand[_] = cards[card_ptr]
                    card_ptr += 1
                state[PLAYER_CARDS[player]] = hand
            except IndexError:
                traceback.print_exc()
                breakpoint()
        state[CARD_PTR] = card_ptr

    def load_game(self, _obs, _info):
        if self.debug:
            print(f"FastTockGame.load_game: Loading game from obs, info")
            print(f"obs : {_obs}")
            print(f"info: {_info}")
        obs = copy.deepcopy(_obs)
        info = copy.deepcopy(_info)
        state = self.state
        state[ROUND] = info['round'] - 1
        state[CURRENT_PLAYER] = info['player']
        self.load_cards(info['player'], info['cards'][:])
        absloc_t = ABSLOC_TABLE
        obs = [sorted(obs[i]) for i in range(NPLAYERS)]
        for player in range(NPLAYERS):
            state[RELLOCS[player]] = obs[player] 
            for pawn in range(4):
                if obs[player][pawn] in [0, BOARD_LEN + 1]:
                    continue
                state[PL_ABSL_PTR + player] |= (1 << absloc_t[player][obs[player][pawn]])
            state[ABSLOCS] |= state[PLAYER_ABSLOCS][player]
        curr_action = info['action']
        if curr_action == None:
            state[CURRENT_ACTION] = PREVIOUS_FOLD
        elif curr_action == False:
            state[CURRENT_ACTION] = NEW_GAME
        else:
            state[CURRENT_ACTION] = curr_action
        # self._check_bits()
        self._update_action_space()
        if self.debug:
            print(f"Game state is now:")
            print(self)
        return self._get_gamestate()

    def load_full(self, game):
        if self.debug:
            print(f"FastTockGame.load_full: Loading full tock game from game:")
            print(game)
        obs, rew, done, info = game._get_gamestate()
        self.load_game(obs, info)
        state = self.state
        state[CARDS] = game.deck.cards[:]
        state[CARD_PTR] = game.deck.pointer
        for player in range(len(obs)):
            env_cards = game.players[player].cards[:]
            while len(env_cards) < 5:
                env_cards.append(NOCARD)
            state[PLAYER_CARDS[player]] = env_cards
        self.rng = copy.deepcopy(game.deck.rng)
        if self.debug:
            print(f"Game state is now:")
            print(self)
    
    def rewind(self):
        if self.debug:
            print(f"FastTockGame.rewind: Resetting game to move 0 from move {self.state[MOVE_NUMBER]}")
        self.state = self.history[0][:]
        if self.debug:
            print(f"Rewound game state to:")
            print(self)

    def load_cards(self, player, cards):
        while(len(cards)) < 5:
            cards.append(NOCARD)
        assert len(cards) == 5
        self.state[PLAYER_CARDS[player]] = cards

    def print_visible(self):
        state = self.state
        rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        cards = [self.get_cards_list(player) for player in range(NPLAYERS)]
        s  = '=======================================\n'
        s += '        Current Fast Game State        \n'
        s += '=======================================\n'
        for player in range(NPLAYERS):
            if player == self.state[CURRENT_PLAYER]:
                card_str = ', '.join([str(CARD_TO_REPR[cards[player][i] % 13]) for i in range(len(cards[player]))])
                s += f"\033[1mPlayer {player}: locs {rellocs[player]}, cards [{card_str}] (current)\033[0m\n"
            else:
                s += (f"Player {player}: locs {rellocs[player]}, cards [?]\n")
        s += '=======================================\n'    
        print(s, end='')   


    def __repr__(self) -> str:
        state = self.state
        rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        cards = [self.get_cards_list(player) for player in range(NPLAYERS)]
        s  = '=======================================\n'
        s += '        Current Fast Game State        \n'
        s += '=======================================\n'
        for player in range(NPLAYERS):
            card_str = ', '.join([str(CARD_TO_REPR[cards[player][i] % 13]) for i in range(len(cards[player]))])
            if player == self.state[CURRENT_PLAYER]:
                s += f"\033[1mPlayer {player}: locs {rellocs[player]}, cards [{card_str}] (current)\033[0m\n"
            else:
                s += (f"Player {player}: locs {rellocs[player]}, cards [{card_str}]\n")
        s += '=======================================\n'    
        return s    



# =============================== Functions ======================================
from tock import make
import copy

def divergence_test(env, fastenv):
    try:
        obs, rew, done, info = env._get_gamestate()
        obs1, rew1, done1, info1 = fastenv._get_gamestate()
        info1['space'] = action_mask_to_tuple_action_space(info1['space'], obs[info1['player']], info1['cards'])
        if info1['action']:
            # info1['action'] = obs_shift_action_tuple(info['action'], obs[info1['player']], reverse=False)
            info1['action'] = info['action']
        diverted = False
        state = fastenv.state
        sorted_obs = [sorted(obs[i]) for i in range(len(obs))]
        if sorted_obs != obs1:
            print(f"(sorted) obs are not equal")
            print(f"{sorted_obs}\n{obs1}")
            diverted = True
        if rew != rew1 or done != done1:
            print(f"rew/done are not equal")
            diverted = True

        if info != info1:
            print(f"infos are not equal")
            print(f"{info}\n{info1}")
            diverted = True
        if state[CARDS] != env.deck.cards or state[CARD_PTR] != env.deck.pointer:
            print(f"Decks are not equal")
            print(f"Deck cards:\n{env.deck.cards}\n{state[CARDS]}")
            print(f"Deck pointers:\n{env.deck.pointer}\n{state[CARD_PTR]}")
            diverted = True

        if diverted:
            print(env)
            print(fastenv)
            # print(fastenv.history)
            print("Games diverted")
            breakpoint()
            return False
        return True
    except Exception:
        traceback.print_exc()
        breakpoint()

def test_fastenv(ngames, nplayers, unmove=True, startmove = 0):
    check_rnd_str = ""
    print("Testing FastTockGame correctness.")
    checkpoint = ngames // 10
    for game in range(ngames):
        if game % checkpoint == checkpoint - 1:
            print(f"Now doing game {game+1}")
        envactions = []
        action_numbers = []
        gamehis = []
        env = make(nplayers)
        obs, rew, done, info = env.reset()
        for i in range(startmove):
            action = choice(info['space'])
            obs, rew, done, info = env.step(action)
        fastenv = FastTockGame(nplayers, max_his_len=100*nplayers, store_rng_his=True)
        fastenv.load_full(env)
        obs, rew, done, info = env._get_gamestate()
        obs1, rew1, done1, info1 = fastenv.load_game(obs, info)
        move = 0
        while not done:
            divergence_test(env, fastenv)
            action_mask = fastenv.get_action_space()
            legal_action_numbers = np.arange(50)[action_mask]
            action_number = choice(legal_action_numbers)
            # action = choice(info['space'])
            player = info1['player']
            cards = info1['cards']
            action = action_number_to_action_tuples(action_number, obs1[player], cards)[-1]
            envaction = obs_shift_action_tuple(action, obs[player], reverse=True)

            if envaction not in info['space']:
                print(env)
                print(fastenv)
                print(f"Action {envaction} not in space {info['space']}")
                breakpoint()
            gamehis.append(copy.deepcopy(env))
            # action_number = action_tuple_to_action_number(action, obs1[player], cards)
            # action_number1 = action_tuple_to_action_number(envaction, obs[player], cards)
            # if action_number != action_number1:
            #     print(f"Action numbers not equal: {action_number} != {action_number1}")
            #     _as = np.zeros(50, dtype=bool)
            #     _as[action_number] = True
            #     _as[action_number1] = True
            #     print_action_space(_as)
            #     breakpoint()
            action_numbers.append(action_number)
            envactions.append(envaction)
            obs1, rew1, done1, info1 = fastenv.step(action=action_number)
 
            # actions.append(action)
            envactions.append(envaction)
            obs, rew, done, info = env.step(action=envaction)
            # print(env)
            # print(fastenv)
            # print(fastenv.history)
            fastenv_copy = copy.deepcopy(fastenv)
            for i in range(move, -1, -1):
                # print_block(f"Unmoving fastenv")
                fastenv_copy.unmove()
                divergence_test(gamehis[i], fastenv_copy)
            for i in range(move):
                # print_block(f"Removing fastenv")
                fastenv_copy.step(action_numbers[i])
                divergence_test(gamehis[i+1], fastenv_copy)
            move += 1
    print("Passed the test!")

def manual_test():
    try:
        env = FastTockGame(nplayers=2)
        obs, rew, done, info = env.reset(2, 100)
        while not done:
            while True:
                print(env)
                print(info)
                print(env.history)
                q = input("\n--- Press 'r' to rewind ---")
                if q == 'r':
                    env.unmove()
                else:
                    break
            action = choice(info['space'])
            obs, rew, done, info = env.step(action=action)
        print("Game ended in position:")
        print(env)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        breakpoint()

def speed_test(ngames, nplayers, deepcopy=False):
    print(f"Speed testing the fast game environment vs. regular, nplayers={nplayers}, ngames={ngames}")
    for env in [FastTockGame(nplayers = 2), make(nplayers=nplayers)]:
        start = time()
        his = []
        for game in range(ngames):
            obs, rew, done, info = env.reset()
            while not done:
                if deepcopy and hasattr(env, "current_action"):
                    his.append(copy.deepcopy(env))
                action = choice(info['space'])
                obs, rew, done, info = env.step(action=action)
        duration = time() - start
        print(f"Took {duration} seconds")

def main():
    game = FastTockGame(nplayers=2)
    obs, rew, done, info = game._get_gamestate()
    for i in range(10):
        action_numbers = np.arange(50)[info['space']]
        obs, rew, done, info = game.step(action_numbers[0])
        _,_,_, info2 = game._get_gamestate(gettuple_as=True)
        order = [2, 3, 1, 0]
        player_obs = obs[info['player']]
        scrambled_obs = [player_obs[order[i]] for i in range(4)]
        scrambled_space = [(card, order.index(pawn)) for card, pawn in info2['space']]
        action_mask = tupled_action_space_to_action_mask(scrambled_space, scrambled_obs, info['cards'])
        print_action_space(action_mask)
        print_action_space(info['space'])
        assert np.equal(action_mask, info['space']).all()
    # manual_test()
    # test_fastenv(100, nplayers=2)
    # speed_test(1000, 2, deepcopy=True)
    # fastenv = FastTockGame(2)
    # obs, rew, done, info = fastenv.reset()
    # move = 0
    # while not done:
    #     valid_actions = np.arange(50)[info['space']]
    #     obs, rew, done, info = fastenv.step(choice(valid_actions))
    #     action_prob = np.abs(np.random.randn(50))
    #     action_prob[info['space']] /= np.sum(action_prob[info['space']])
    #     action_tuples, action_prob = action_mask_to_tuple_action_space(info['space'], obs[info['player']], info['cards'], action_probabilities=action_prob, verbose=True)
    #     move += 1


if __name__ == "__main__":
    main()
