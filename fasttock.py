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

# ================================= Classes ===========================================

ACTION_TUPLES = [(0, 0) for i in range(20)]
for action_number in range(20):
    cardidx, pawn = ACTION_TUPLES[action_number] = (action_number // 4, action_number % 4)




class StateWatchdog(list):
    def __setitem__(self, key, value):
        # Key can be an int or a slice object
        print(f"DEBUG: state[{key}] changed to {value}")
        
        # Optional: Print the traceback to see WHICH line of code did it
        import traceback; traceback.print_stack()
        
        super().__setitem__(key, value)


def set_globals(nplayers=2):
    global ALL_BITMASKS, PLAYER_BITMASKS, POS_MASKS, BOARD_LEN, NPLAYERS,\
            CARD_TO_VALUE, KING, ACE, PLAYER, ROUND, OFFSETS, CURRENT_PLAYER,\
            ABSLOCS, PLAYER_ABSLOCS, RELLOCS, PLAYER_CARDS, CARDS, CARD_PTR,\
            ACTION_SPACE, GAMESPACE_LEN, CARD_TO_REPR, DECK_LEN, PL_ABSL_PTR,\
            RELLOCS_PTR, PLAYER_CARDS_PTR, PREVIOUS_FOLD, NEW_GAME, CURRENT_ACTION,\
            MOVE_NUMBER, RELLOC_TABLE, ABSLOC_TABLE, ABSLOC_VALUE_TABLE, RELLOC_START_PTR,\
            ACTION_TUPLES, MAX_HIS_LEN, PLAYER_ALL_CARDS, NOCARD, ALL_RELLOCS, PREVIOUS_PLAY
    NPLAYERS = nplayers
    BOARD_LEN = NPLAYERS * 16
    DECK_LEN = NPLAYERS * 13
    NOCARD = DECK_LEN
    MAX_HIS_LEN = NPLAYERS * 100
    OFFSETS = [player * 16 for player in range(NPLAYERS)]
    CARD_TO_VALUE = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1] * NPLAYERS
    CARD_TO_REPR = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "H", "A"]
    PREVIOUS_FOLD = 20
    NEW_GAME = 21
    PREVIOUS_PLAY = 22

    KING = 0
    ACE = 1

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
    CURRENT_ACTION = 6 + 23*NPLAYERS
    MOVE_NUMBER = 7 + 23*NPLAYERS

    GAMESPACE_LEN = 8 + 23*NPLAYERS


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
    def __init__(self, nplayers=2, max_his_len=None, store_rng_his=False):
        set_globals(nplayers=nplayers)
        self.rng = np.random.default_rng()
        # self.state = StateWatchdog([0] * GAMESPACE_LEN)
        self.store_rng_his = store_rng_his
        self.reset(nplayers=nplayers, max_his_len=max_his_len)
    
    def get_hash(self):
        return (tuple(self.state[PLAYER_ABSLOCS]), tuple(self.state[PLAYER_ALL_CARDS]))

    def get_player_hash(self):
        state = self.state
        current_player = state[CURRENT_PLAYER]
        cards =  self.get_cards_list(current_player)
        cards.sort()
        current_action = state[CURRENT_ACTION]
        if current_action not in [NEW_GAME, PREVIOUS_FOLD]:
            current_action = PREVIOUS_PLAY
        return (tuple(state[ALL_RELLOCS]), tuple(cards),
                current_player, current_action, state[ROUND])

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
        # check if pawn makes it to homebase
        # pawn can not be blocked by other player because value < LENGTH_SEGMENT
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
                if card_value in [KING, ACE] :  # heer of aas
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
                if card_value == KING:
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

    def step(self, action):
        state = self.state
        converted_action = self.convert_action(action)
        
        """
        performs action on the current player and returns the gamestate after the action
        if action is not valid, a random action from the possible actions is executed
        """

        # if action is not valid, take a random valid action
        if not (state[ACTION_SPACE] >> self.get_action_number(converted_action)) & 1:
            traceback.print_exc()
            print("Error: action not in action space")
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
        if self.store_rng_his:
            self.rng_his.append(self.rng.bit_generator.state)
        state[CURRENT_ACTION] = self.get_action_number(action) # store unconverted, isn't used anyways
        state[MOVE_NUMBER] += 1
        state[CURRENT_ACTION] = self.get_action_number(action)
        cardidx, pawn = converted_action
        player = state[CURRENT_PLAYER]
        card = state[PLAYER_CARDS[player]][cardidx]
        value = CARD_TO_VALUE[card]
        self._execute_move(pawn, value)
        # print(f"Player {state[CURRENT_PLAYER]} executed move {(CARD_TO_REPR[card % 13], pawn)} successfully")
        state[PLAYER_CARDS_PTR + 5*player + cardidx] = NOCARD

        # determine the next player 
        self._get_next_player() 

        return self._get_gamestate()


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
                card_list.append(card)
        return card_list
    
    def convert_action(self, action):
        card, pawn = action
        cards = self.state[PLAYER_CARDS[self.state[CURRENT_PLAYER]]]
        for i in range(5):
            if cards[i] == NOCARD:
                card += 1
            if i == card:
                return (i, pawn)
        print(f"Invalid action card index for action {action} with cards {cards}")
        breakpoint()

    def get_action_space(self):
        state = self.state
        action_space = state[ACTION_SPACE]
        non_converted = [ACTION_TUPLES[i] for i in range(20) if ((action_space >> i) & 1)]
        cards = state[PLAYER_CARDS[state[CURRENT_PLAYER]]]
        nocard_count = 0
        converted_cards = list(range(5))
        for i in range(5):
            if cards[i] == NOCARD:
                nocard_count += 1
            converted_cards[i] = i - nocard_count
        converted = [(converted_cards[non_converted[i][0]], non_converted[i][1]) for i in range(len(non_converted))]
        return converted
    
    def _get_gamestate(self):
        state = self.state
        current_player = state[CURRENT_PLAYER]
        curr_act_number = state[CURRENT_ACTION]
        if curr_act_number == NEW_GAME:
            curr_act = False
        elif curr_act_number == PREVIOUS_FOLD:
            curr_act = None
        else:
            curr_act = ACTION_TUPLES[curr_act_number]
        rew = None
        done, winner = self._done()
        obs = rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        info = {
                'player': current_player,
                'cards' : self.get_cards_list(current_player),
                'space' : self.get_action_space(),
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
                    f"Error: cannot undo more than {len(self.history)} moves,"
                    f"reset with appropriate 'max_his_len' to increase buffer size"
            )
            breakpoint()
        if self.store_rng_his:
             self.rng.bit_generator.state = self.rng_his.pop()
        self.state = self.history[old_move_number][:]

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
        # print("---------------------")
        # print("Resetting game state!")
        # reset all players
        self.state = [0] * (GAMESPACE_LEN)
        self.history = [[0]*GAMESPACE_LEN for i in range(max_his_len)]
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
        # print("Reset game state successfully")
        # print("---------------------")
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

    def load_game(self, obs, info):
        state = self.state
        state[ROUND] = info['round'] - 1
        state[CURRENT_PLAYER] = info['player']
        self.load_cards(info['player'], info['cards'][:])
        absloc_t = ABSLOC_TABLE
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
            state[CURRENT_ACTION] = self.get_action_number(curr_action)
        # self._check_bits()
        self._update_action_space()
        return self._get_gamestate()

    def load_full(self, game):
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
        if self.store_rng_his:
            self.rng = copy.deepcopy(game.deck.rng)
    
    def rewind(self):
        self.state = self.history[0][:]

    def load_cards(self, player, cards):
        while(len(cards)) < 5:
            cards.append(NOCARD)
        assert len(cards) == 5
        self.state[PLAYER_CARDS[player]] = cards

    def __repr__(self) -> str:
        state = self.state
        rellocs = [state[RELLOCS[i]] for i in range(NPLAYERS)]
        cards = [self.get_cards_list(player) for player in range(NPLAYERS)]
        s  = '=======================================\n'
        s += '        Current Fast Game State        \n'
        s += '=======================================\n'
        s += f"     Round         : {state[ROUND]}\n"
        s += f"     Current player: {state[CURRENT_PLAYER]}\n"
        s += f"     Current move  : {state[MOVE_NUMBER]}\n"
        for player in range(NPLAYERS):
            card_str = [CARD_TO_REPR[cards[player][i] % 13] for i in range(len(cards[player]))]
            s += (f"Player {player}: locs {rellocs[player]}, cards {card_str}\n")
        # action_space = state[ACTION_SPACE]
        # actions = self.get_action_space()
        # s += (f"Action space: {actions}\n")
        # s += (f"Deck: {state[CARDS]}\n")
        # s += (f"Card ptr: {state[CARD_PTR]}\n")
        # s += self._print_abslocs()
        s += '=======================================\n'    
        return s    



# =============================== Functions ======================================
from tock import make
import copy

def divergence_test(env, fastenv):
    obs, rew, done, info = env._get_gamestate()
    obs1, rew1, done1, info1 = fastenv._get_gamestate()
    diverted = False
    state = fastenv.state
    if obs != obs1:
        print(f"obs are not equal")
        print(f"{obs}\n{obs1}")
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

def test_fastenv(ngames, nplayers, unmove=True):
    check_rnd_str = ""
    print("Testing FastTockGame correctness.")
    checkpoint = ngames // 10
    for game in range(ngames):
        if game % checkpoint == checkpoint - 1:
            print(f"Now doing game {game+1}")
        actions = []
        gamehis = []
        env = make(nplayers)
        obs, rew, done, info = env.reset()
        for i in range(15):
            action = choice(info['space'])
            obs, rew, done, info = env.step(action)
        fastenv = FastTockGame(nplayers, max_his_len=100*nplayers, store_rng_his=True)
        fastenv.load_full(env)
        obs, rew, done, info = env._get_gamestate()
        obs1, rew1, done1, info1 = fastenv.load_game(obs, info)
        move = 0
        while not done:
            divergence_test(env, fastenv)
            action = choice(info['space'])
            actions.append(action)
            gamehis.append(copy.deepcopy(env))
            obs, rew, done, info = env.step(action=action)
            obs1, rew1, done1, info1 = fastenv.step(action=action)
            # print(env)
            # print(fastenv)
            # print(fastenv.history)
            fastenv_copy = copy.deepcopy(fastenv)
            for i in range(move, -1, -1):
                fastenv_copy.unmove()
                divergence_test(gamehis[i], fastenv_copy)
            for i in range(move):
                fastenv_copy.step(actions[i])
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
    # manual_test()
    # test_fastenv(10, nplayers=2)
    speed_test(1000, 2, deepcopy=True)


if __name__ == "__main__":
    main()
