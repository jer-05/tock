from random import choice
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
import traceback
import inspect
import torch.multiprocessing as mp

from tock import PLACES_PER_SEGMENT, DEAL_ORDER
from tock import TockGame, make
from utils import getstate, totorch, get_card_str, clean_fname, get_ort_inference_session, get_action_string, print_block, print_important
from mcts import MCTS, Node
from data import GameData
from fasttock import FastTockGame, return_search_results, CTV
from network import PolicyNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def get_fname(player_names, player_configs, filedir, extension="", extra_info=""):
    data_str = ""
    for name, cfg in zip(player_names, player_configs):
        cfg = clip_cfg(cfg, del_long=True)
        data_str += f"{name}_"
        for key, value in cfg.items():
            data_str += f"{key}-{value}_"
    if extra_info:
        data_str = f"{data_str}{extra_info}"
    else:
        data_str = data_str[:-1]
    fname = f"{data_str}{extension}"    
    fname = clean_fname(fname)
    return f"{filedir}/{fname}"



def get_fullnames(player_names, player_configs):
    fullnames = []
    for name, cfg in zip(player_names, player_configs):
        if len(cfg.items()) == 0:
            fullnames.append(name)
            continue
        fullname = f"{name}["
        for i, (key, value) in enumerate(cfg.items()):
            fullname += f"{key}:{value}"
            if i != len(cfg.items()) - 1:
                fullname += ", "
        fullname += "]"
        fullnames.append(fullname)
    return fullnames


def clip_cfg(config, del_long=False):
    clipped_cfg = {}
    for key, value in config.items():
        if len(f"{value}") < 15:
            clipped_cfg[key] = value
        elif key == "fname":
            inv = value[::-1]
            inv = inv[:inv.index("/")]
            clipped_cfg[key] = f"?/{inv[::-1]}"
        elif key == "hyperparams":
            clipped_cfg["hypp"] = clip_cfg(value)
        else:
            if not del_long:
                clipped_cfg[key] = "..."
    return clipped_cfg

def get_players(player_names, player_configs=None):
    if player_configs == None:
        player_configs = [{} for _ in range(len(player_names))]
    players = []
    for player_name, cfg in zip(player_names, player_configs):
        cls, attr = PLAYER_MAP[player_name]
        if cls == None:
            players.append(attr)
        else:
            instance = cls(**cfg)
            players.append(getattr(instance, attr))
    types = []
    for i, player in enumerate(players):
        sig = inspect.signature(player)
        if "info" in sig.parameters.keys() and "obs" in sig.parameters.keys():
            types.append('obsinfo')
        elif "game" in sig.parameters.keys():
            types.append('environment')
        else:
            assert False
    return players, types

class NNPlayer():
    def __init__(self, fname=None, gpu_manager_info=None, worker_id=None, ort_info=None, hyperparams={}, return_type = "best_action", nplayers = 2, verbose=False):
        self.gpu_manager_info = gpu_manager_info
        if not gpu_manager_info:
            self.nn = PolicyNN(nplayers=nplayers)
            if fname != None:
                self.nn.load(fname, silent=True)
            self.ort_session = get_ort_inference_session(self.nn, fname, worker_id, *ort_info)
        else:
            self.nn = None
            self.use_gpu_manager = True
        self.hyperparams = hyperparams
        self.return_type = return_type
        self.worker_id = worker_id
        self.verbose = verbose
        self.nplayers = nplayers

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def mcts_move(self, obs, info):
        mcts = MCTS(obs, info, self.return_type, self.ort_session, hyperparams=self.hyperparams, verbose=self.verbose)
        mcts.iterate()
        if self.verbose:
            mcts.save_tree("figures/mcts/mcts_tree.png", maxdepth=2, max_branching=3)
        return mcts.get_results()

    def virt_mcts_move(self, obs, info):
        only_move_result = only_move(obs, info, return_type = self.return_type, verbose=self.verbose)
        if only_move_result is not None:
            return only_move_result
        actual_info = copy.deepcopy(info)
        vmcts = VMCTS(obs, info, worker_id = self.worker_id, hyperparams=self.hyperparams, gpu_manager_info=self.gpu_manager_info, verbose=self.verbose)
        vmcts.iterate()
        unique_prob, val = vmcts.get_results()
        if self.verbose:
            self.show(obs, info)
            vmcts.save_tree("figures/mcts/mcts_tree", maxdepth=2, max_branching=3)
        prob = deuniqueify_prob(actual_info['space'], actual_info['cards'], unique_prob)
        if self.return_type == "probabilities":
            return prob
        return actual_info['space'][np.argmax(prob)]

    def move(self, obs, info):
        cards = info['cards']
        state = getstate(obs, info, getdict=True, fastgame_type=False)
        logits, value_head = self.ort_session.run(None, state)
        logits = logits[0]
        if self.verbose:
            print(f"NNPlayer: Evaluating position:")
            game = FastTockGame(nplayers=len(obs))
            game.load_game(obs, info)
            game.print_visible()
            print(f"Value head: ",  f'{value_head.item():.3f}' if len(obs) == 2 else ', '.join([format(v, ':3f') for v in value_head[0]]))
        policy_head = np.exp(logits) / np.sum(np.exp(logits))
        return return_search_results(self.return_type, state["action_mask"][0].astype(bool), obs[info['player']], info, policy_head, verbose=self.verbose, check_normalization=True, tau=None)

CARD_TO_VALUE_HAS_STARTER = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 15]
MEAN_CARD_VALUE =  sum(CARD_TO_VALUE_HAS_STARTER) / 13
START_PENALTY = 15
CARD_TO_VALUE = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1]
MEAN_CARD_DISTANCE = np.mean(CARD_TO_VALUE)
MAX_NORMALIZATION = 4
MIN_NORMALIZATION = 0.4
PAWN_ON_FIRST_BONUS = 10

def symmetric_eval_board(obs, info, verbose=False):
    """
    Return an array with winning probabilities for each player
    """
    nplayers = len(obs)
    current_player = info['player']
    board_len = nplayers * PLACES_PER_SEGMENT
    base_bonus = 2 * board_len
    four_bonus = 1.5 * board_len
    useable_fours = 0
    current_start_pawns = 0
    scores = np.zeros(nplayers)

    cards = info['cards']
    ncards = len(cards)
    cards = [cards[i] % 13 for i in range(ncards)]
    best_card_value = max([CARD_TO_VALUE[card] for card in cards])
    max_card_score = 0

    # position scores
    for player in range(nplayers):
        for pawn in range(4):
            loc = obs[player][pawn]
            if loc == 0:
                scores[player] -= START_PENALTY
                if player == current_player:
                    current_start_pawns += 1
            elif loc == board_len + 1:
                scores[player] += base_bonus
            elif loc == 1:
                scores[player] += PAWN_ON_FIRST_BONUS
            else:
                scores[player] += loc
                if player == current_player:
                    max_card_score += board_len - loc
                    if loc <= 4:
                        useable_fours += 1
                    if loc > board_len - best_card_value:
                        scores[player] += loc
                        continue
                if loc > board_len - 2 * MEAN_CARD_DISTANCE:
                    scores[player] += .5 * loc

    # compute the potential value of cards in hand; for opponents assume they
    # have standard cards (since their cards are not known)
    card_score = 0
    fours = 0
    for card in cards:
        if current_start_pawns and card in [11, 12]:
            max_card_score += board_len
            useable_fours += 1
            current_start_pawns -= 1
            card_score += START_PENALTY
        else:
            card_score += CARD_TO_VALUE[card]
            if card == 2:
                fours += 1
    used_fours = 0
    card_score = min(card_score, max_card_score)

    # add bonus points for having useable fours (which are very valuable)
    four_bonus_total = 0
    while fours > 0:
        if useable_fours > 0:
            four_bonus_total += four_bonus
            useable_fours -= 1
            used_fours += 1
        fours -= 1
    card_scores = np.zeros(nplayers)
    opp_card_score = MEAN_CARD_VALUE * (ncards - 0.5)
    if info['action'] == None and DEAL_ORDER[info['round']%3 - 1] != ncards:
        opp_card_score *= (nplayers - 2) / (nplayers - 1)
    for player in range(nplayers):
        if player == current_player:
            card_scores[player] = card_score
        else:
            card_scores[player] = opp_card_score

    if verbose:
        print(f"Evaluating position (d0 eval):")
        print(f"  Current player                     : {current_player}")
        print(f"  Scores (board — cards — four bonus):")
        for player, (board_score, card_score) in enumerate(zip(scores, card_scores)):
            if player != current_player:
                print(f"\tPlayer {player}: {board_score:.0f} — {card_score:.1f}")
            else:
                print(f"\tPlayer {player}: {board_score:.0f} — {card_score:.1f} — {four_bonus_total}")
    scores += card_scores
    scores[current_player] += four_bonus_total

    # rotate scores to player's perspective and normalize
    scores_from_player_perspective = np.array([scores[(i + current_player) % nplayers] for i in range(nplayers)])

    # normalize scores. If the game is close to finishing, probabilities become sharper with score differences.
    max_score = np.max(scores_from_player_perspective)
    normalization_modifier = np.clip(MAX_NORMALIZATION + (MIN_NORMALIZATION - MAX_NORMALIZATION) * max_score / (base_bonus * 4 + 25), MIN_NORMALIZATION, MAX_NORMALIZATION)
    normalization_factor = normalization_modifier * board_len
    normalized_scores = scores_from_player_perspective / normalization_factor
    winning_chances = np.exp(normalized_scores) / np.sum(np.exp(normalized_scores))
    if verbose:
        print(f"  Normalization: modifier {normalization_modifier:.2f}, factor {normalization_factor:.0f}")
        print(f"  Total Score, Normalized Score, Winning Probability (rel. to player {current_player}):")
        for player, (score, norm_score, chance) in enumerate(zip(scores_from_player_perspective, normalized_scores, winning_chances)):
            print(f"\tRel. player {player}: {score:.1f}, {normalized_scores[player]:.3f}, {chance*100:.1f}%")
    return winning_chances

EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

def only_move(obs, info, *, return_type="best_action", verbose=False):
    unique_as = get_unique_actions(info['space'], info['cards'])
    if len(unique_as) == 1:
        if verbose:
            action = info['space'][0]
            print(f"Only move detected: {get_action_string(action, info['cards'])}")
        if return_type in ["best_action", "best_move"] or return_type is None:
            if verbose:
                print(f"Returning move")
            return info['space'][0]
        elif return_type in ["probabilities", "norm_scores"]:
            nactions = len(info['space'])
            ret =  np.array([1 / nactions] * nactions)
            if verbose:
                print(f"Returning prob/evals {ret}")
            return ret
        else:
            print(f"only_move: got unkown return type {return_type}")
            breakpoint()
    return None

class TreeSearchPlayer:
    def __init__(self, *, maxdepth=3, evalfunc=symmetric_eval_board, maxit=50, tau =1, return_type="best_action", store_rng=True, alpha_beta_pruning = True, cache_enabled = True, verbose=False):
        self.maxdepth = maxdepth
        self.eval = evalfunc
        self.verbose = verbose
        self.maxit = maxit
        self.return_type = return_type
        self.tau = tau
        self.store_rng = store_rng
        self.alpha_beta_pruning = alpha_beta_pruning
        self.hits = 0
        self.lookups = 0
        self.cache_enabled = cache_enabled

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def tree_search(self, game, depth, alpha, beta):
        """
        Use minimax with transposition tables and pruning to get the 
        evaluation of a position.
        """
        self.lookups += 1
        entry = None
        # check if the current position was already evaluated before
        if self.cache_enabled:
            game_hash = game.get_hash()
            if game_hash in self.cache:
                self.hits += 1
                if self.alpha_beta_pruning:
                    entry = self.cache[game_hash]
                    # do not use the entry if it is based on a shallow search
                    if entry['depth'] > depth:
                        entry = None
                    elif entry['type'] == EXACT:
                        return entry['value']
                    elif entry['type'] == LOWER_BOUND:
                        alpha = max(entry['value'][0], alpha)
                        if alpha > beta:
                            return entry['value']
                    elif entry['type'] == UPPER_BOUND:
                        beta = min(entry['value'][0], beta)
                        if alpha > beta:
                            return entry['value']
                else:
                    return self.cache[game_hash]

        obs, rew, done, info = game._get_gamestate()
        current_player = info['player']
        nplayer = len(obs)
        alpha_orig = alpha

        # return if game state is terminal or maximum search depth is reached
        if info['winner'] != -1:
            vals = np.zeros(nplayer)
            vals[(info['winner'] - current_player) % nplayer] = 1
            return vals
        if depth >= self.maxdepth:
            return self.eval(obs, info)

        # compute the winning chances ("norm_scores") for each position that 
        # arises after taking an action
        action_numbers = np.arange(50)[info['space']]
        norm_scores = -np.inf * np.ones((len(action_numbers), nplayer))

        # check if the position admits an only move
        if depth == 0 and len(action_numbers) == 1:
            return None
        for i, action_number in enumerate(action_numbers):
            newobs, rew, done, newinfo = game.step(action_number)
            new_current_player = newinfo['player']
            if new_current_player == current_player:
                norm_score = self.tree_search(game, depth+1, alpha, beta)
                game.unmove()
            else:
                if nplayer == 2:
                    norm_score = self.tree_search(game, depth+1, 1-beta, 1-alpha)
                else:
                    norm_score = self.tree_search(game, depth+1, -np.inf, 1-alpha)

                game.unmove()
            player_diff = current_player - new_current_player
            for j, val in enumerate(norm_score):
                norm_scores[i][(j - player_diff) % nplayer] = val
            if norm_scores[i][0] > alpha:
                alpha = norm_scores[i][0]
            if self.alpha_beta_pruning:
                if alpha >= beta:
                    break

        # Get the row which corresponds to the entry maximizing the score
        # for the current player (column 0)
        best_norm_score = norm_scores[np.argmax(norm_scores[:, 0])]
        # Store the position evaluation in cache
        if self.cache_enabled:
            if self.alpha_beta_pruning:
                new_entry = {
                        'value': best_norm_score,
                        'depth': depth,
                }
                if best_norm_score[0] <= alpha_orig:
                    new_entry['type'] = UPPER_BOUND
                elif best_norm_score[0] >= beta:
                    new_entry['type'] = LOWER_BOUND
                else:
                    new_entry['type'] = EXACT
                self.cache[game_hash] = new_entry
            else:
                self.cache[game_hash] = best_norm_score

        if depth == 0:
            # We want all scores (for calculating move probabilities) 
            return norm_scores
        return best_norm_score

    def show_search_stats(self, game):
        obs, rew, done, info = game._get_gamestate()
        if self.verbose:
            print_block("TreeSearchPlayer Search Info")
            print(f"    Evaluation function: {self.eval.__name__}")        
            print(f"    Lookups            : {self.lookups}")
            print(f"    Cache hit ratio    : {self.hits / self.lookups * 100:.1f}%")
            if isinstance(game, TockGame):
                print(game)
            else:
                print(game.print_visible())
        return self.eval(obs, info, verbose=self.verbose)

    def move(self, game):
        obs, rew, done, info = game._get_gamestate()
        obs_player = obs[info['player']]
        self.hits = 0
        self.lookups = 0
        self.cache = {}
        if self.store_rng:
            fastgame = FastTockGame(len(obs), max_his_len=self.maxdepth+1, store_rng_his=True)
        else:
            fastgame = FastTockGame(len(obs), max_his_len=self.maxdepth+1, store_rng_his=False)
        fastgame.load_full(game)
        norm_scores = self.tree_search(fastgame, 0, -np.inf, np.inf)
        fobs, frew, fdone, finfo = fastgame._get_gamestate()
        action_mask = finfo['space']
        d0_winning_chances = self.show_search_stats(game)
        if norm_scores is None:
            norm_scores = np.array([d0_winning_chances])
            if self.verbose:
                print_important(f"The position has only one legal move! Search was not executed")
        return return_search_results(self.return_type, fastgame.get_action_space(), obs_player, info, norm_scores[:, 0], verbose=self.verbose, check_normalization=False, tau=self.tau)
    
    def determinization_move(self, obs, info):
        self.hits = 0
        self.lookups = 0
        self.cache = {}
        fastgame = FastTockGame(nplayers=len(obs), max_his_len=self.maxdepth+1)
        fastgame.load_game(obs, info)
        # print(f"Actual action space is {info['space']}")
        for i in range(self.maxit):
            # print(f"Entering iteration {i} with state:")
            # print(fastgame)
            partial_norm_score = self.tree_search(fastgame, 0, -np.inf, np.inf)
            if i == 0:
                norm_scores = partial_norm_score
            else:
                norm_scores += partial_norm_score
            if partial_norm_score is None:
                break
            fastgame._shuffle()
            fastgame.deal()
            fastgame.load_cards(info['player'], info['cards'][:])
        if norm_scores is not None:
            norm_scores /= self.maxit
        cards = info['cards']
        obs_player = obs[info['player']]
        d0_winning_chances = self.show_search_stats(fastgame)
        if norm_scores is None:
            norm_scores = np.array([d0_winning_chances])
            if self.verbose:
                print_important(f"The position has only one legal move! Search was not executed")

        return return_search_results(self.return_type, fastgame.get_action_space(), obs_player, info, norm_scores[:, 0], verbose=self.verbose, check_normalization=False, tau=self.tau)

def random_player(obs, info):
    """
    defines a random player: returns a random action from the actionspace
    """
    action = choice(info['space'])
    return action 

def eager_player(obs, info):
    """defines an eager player: returns an action of the most advanced pawn and highest card value"""
    maxloc, maxval = -1, -5
    player = info['player']
    nplayers = len(obs) 
    board_len = nplayers * PLACES_PER_SEGMENT
    for action in info['space']:
        card, pawn = action
        loc = obs[player][pawn]
        val = CTV[(info['cards'][card])]
        if loc >= maxloc and val > maxval:
            maxloc, maxval, bestaction = loc, val, action
    return bestaction

def get_capture(obs, info, threshold=0.7):
    player = info['player']
    nplayers = len(obs)
    board_len = PLACES_PER_SEGMENT * nplayers
    capture_action = False
    def _abs_loc(loc, player):
        return (loc + player * PLACES_PER_SEGMENT - 1) % board_len
    for action in info['space']:
        card, pawn = action
        val = CTV[(info['cards'][card])]
        rel_loc = obs[player][pawn]
        capture_score = -100
        # capture an opponent's pawn
        loc = _abs_loc(rel_loc, player)
        if val == 0:
            val = 1
        new_loc = (loc + val) % board_len
        for opponent in range(len(obs)):
            if opponent == player:
                continue
            for opp_pawn in range(len(obs[opponent])):
                rel_other_loc = obs[opponent][opp_pawn]
                if rel_other_loc < threshold * board_len:
                    continue
                if rel_other_loc in [0, board_len + 1]:
                    continue
                other_loc =  _abs_loc(rel_other_loc, opponent)
                if new_loc == other_loc:
                    # print(f"capture action: {action}")
                    # print(f"captured pawn: ({opponent}, {opp_pawn})")
                    # print(f"new loc of capturing pawn: {new_loc}")
                    # print(f"obs:\n{obs}")
                    # print(f"info\n{info}")
                    # print("-------\n")
                    if rel_other_loc - val > capture_score:
                        capture_score = rel_other_loc - val
                        capture_action = action
    return capture_action

def capture_player(obs, info):
    action = get_capture(obs, info, 0)
    if action:
        return action
    return eager_player(obs, info)


def smart_player(obs, info):
    """
    Decision rules:
    -1. Put pawn in homebase
    0. Move pawn in position 1, 2, 3 or 4 back 4 
    1. Put new pawn on starting position
    2. Move furthest pawn
    """
    player = info['player']
    nplayers = len(obs)
    board_len = PLACES_PER_SEGMENT * nplayers
    # put in homebase
    for action in info['space']:
        card, pawn = action
        val = CTV[(info['cards'][card])]
        rel_loc = obs[player][pawn]
        if val + rel_loc > board_len:
            return action

    # moveback action
    for action in info['space']:
        card, pawn = action
        rel_loc = obs[player][pawn]
        val = CTV[(info['cards'][card])]
        if val == -4:
            if rel_loc in [1, 2, 3, 4]:
                return action

    # capture advanced piece
    capture_action = get_capture(obs, info)
    if capture_action:
        return capture_action
    
    eager_action = eager_player(obs, info)
    card, pawn = eager_action
    val = CTV[(info['cards'][card])]
    if obs[player][pawn] != 0 and val in [1, 2, 3, -4]:
        # move out pawn
        for action in info['space']:
            card, pawn = action
            rel_loc = obs[player][pawn]
            if rel_loc == 0:
                return action
    return eager_action

def main(ort_info):
    nplayers = 2
    game = make(nplayers)
    ts_player = TreeSearchPlayer(verbose=True, return_type="best_action", alpha_beta_pruning = True, maxdepth=5, maxit=100)
    # fname = "weights/2_players/datasize_59981_epochs_10_batch_128_lr_0.001_iteration_1.pth"
    # nn_player = NNPlayer(fname=fname, verbose=True, return_type="best_action", nplayers = nplayers, ort_info=ort_info)
    movefunc = ts_player.move
    # movefunc = nn_player.move
    startmove = 0
    while True:
        move_no = 0
        obs, rew, done, info = game.reset()
        while not done:
            move_no += 1
            if move_no > startmove:
                print(f"\n---------------------------------")
                # res = movefunc(obs, info)
                res = movefunc(game)
                try:
                    cont = input("------ Press Enter To Continue -------")
                except EOFError:
                    print(f"\nRestarting game\n")
                    break
                

            obs, rew, done, info = game.step(res)
        print("Game finished")
        if info['winner'] != -1:
            print(f"Player {info['winner']} won")


if __name__ == '__main__':
  from mp_ort_import import exec_main
  exec_main(main)

PLAYER_MAP  = {"TSPEnv": (TreeSearchPlayer, "move"),
               "TSPDet": (TreeSearchPlayer, "determinization_move"),
               "NN": (NNPlayer, "move"),
               "Smart": (None, smart_player),
               "Eager": (None, eager_player),
               "Random": (None, random_player),
               "Capture": (None, capture_player),
               "NNMCTS": (NNPlayer, "mcts_move"),
               "VNNMCTS": (NNPlayer, "virt_mcts_move"),
               }
