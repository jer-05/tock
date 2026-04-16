from random import choice
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
import traceback
import inspect

from mcts_virtual_loss import VMCTS
from tock import PLACES_PER_SEGMENT, DEAL_ORDER
from tock import TockGame, make
from utils import getstate, get_action_prob, totorch, policy_to_actions, print_evals_and_info, get_card_str, print_evals_and_info, get_unique_actions, deuniqueify_prob, clean_fname
from mcts import MCTS, Node
from data import GameData
from fasttock import FastTockGame
from network import PolicyNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def get_fname(player_names, player_configs, filedir, extension="pth", extra_info=""):
    data_str = ""
    for name, cfg in zip(player_names, player_configs):
        cfg = clip_cfg(cfg, del_long=True)
        data_str += f"{name}_"
        for key, value in cfg.items():
            data_str += f"{key}-{value}_"
    if extra_info:
        data_str += extra_info
    else:
        data_str = data_str[:-1]
    fname = f"{data_str}.{extension}"    
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
        else:
            if not del_long:
                clipped_cfg[key] = "[...]"
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
    def __init__(self, fname=None, gpu_manager_info=None, worker_id=None, hyperparams={}, return_type = "best_action", nplayers = 2, use_tensorflow=False, test_tf_loader=False, verbose=False):
        self.gpu_manager_info = gpu_manager_info
        self.use_tensorflow = use_tensorflow
        if not gpu_manager_info:
            if test_tf_loader:
                self.nn = get_loaded_nn(fname, nplayers)
            else:
                if use_tensorflow:
                    self.nn = TFPolicyNN(nplayers=nplayers)
                else:
                    self.nn = PolicyNN(nplayers=nplayers)
                if fname != None:
                    self.nn.load(fname, silent=True, device='cpu')
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
        if len(info['space']) == 1:
            if self.return_type in ["eval", "probabilities"]:
                return np.array([1])
            return info['space'][0]
        actual_info = copy.deepcopy(info)
        mcts = MCTS(obs, info, self.nn, hyperparams=self.hyperparams, use_tensorflow = self.use_tensorflow, verbose=self.verbose)
        mcts.iterate()
        unique_prob, val = mcts.get_results()
        if self.verbose:
            self.show(obs, info)
            mcts.save_tree("figures/mcts/mcts_tree.png", maxdepth=2, max_branching=3)
        prob = deuniqueify_prob(actual_info['space'], actual_info['cards'], unique_prob)
        if self.return_type in ["eval", "probabilities"]:
            return prob
        return actual_info['space'][np.argmax(prob)]

    def virt_mcts_move(self, obs, info):
        if len(info['space']) == 1:
            if self.return_type in ["eval", "probabilities"]:
                return np.array([1])
            return info['space'][0]
        actual_info = copy.deepcopy(info)
        vmcts = VMCTS(obs, info, worker_id = self.worker_id, hyperparams=self.hyperparams, gpu_manager_info=self.gpu_manager_info, verbose=self.verbose)
        vmcts.iterate()
        unique_prob, val = vmcts.get_results()
        if self.verbose:
            self.show(obs, info)
            vmcts.save_tree("figures/mcts/mcts_tree", maxdepth=2, max_branching=3)
        prob = deuniqueify_prob(actual_info['space'], actual_info['cards'], unique_prob)
        if self.return_type in ["eval", "probabilities"]:
            return prob
        return actual_info['space'][np.argmax(prob)]

    def infere(self, obs, info):
        if self.use_tensorflow:
            logits, value_head = self.nn(getstate(obs, info, gettf=True))
            policy_head = tf.nn.softmax(logits, axis=1)[0].numpy()
        else:
            self.nn.eval()
            with torch.no_grad():
                logits, value_head = self.nn(*getstate(obs, info, gettorch=True))
                policy_head = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        assert np.isclose(np.sum(policy_head), 1)
        if self.nplayers == 2:
            return policy_head, value_head.numpy()
        else:
            if self.use_tensorflow:
                value_head = tf.nn.softmax(value_head, axis=1)[0].numpy()
            else:
                value_head = torch.softmax(value_head, dim=1)[0].detach().cpu().numpy()
            return policy_head, value_head

    def move(self, obs, info):
        info['space'] = get_unique_actions(info['space'], info['cards'])
        policy_head, value_head = self.infere(obs, info)
        action_prob = get_action_prob(policy_head, info)
        if self.verbose:
            self.show(obs, info)
        if self.return_type == "eval":
            nplayers = len(obs)
            if nplayers == 2:
                return np.array([value_head, -value_head])
            else:
                return value_head
        best_action = info['space'][np.argmax(action_prob)]
        return best_action

    def show(self, obs, info):
        self.nn.eval()
        info['space'] = get_unique_actions(info['space'], info['cards'])
        policy_head, value_head = self.infere(obs, info)
        action_prob = get_action_prob(policy_head, info)
        preferred_actions, print_actions = policy_to_actions(policy_head, info['cards'])
        invalid_preferences_str = ""
        invalid_preferences = 0
        for i, action in enumerate(preferred_actions):
            if action in info['space']:
                break
            invalid_preferences_str += f"  {i+1}: ({print_actions[i][0]}, {print_actions[i][1]})\n"
            invalid_preferences += 1
        if invalid_preferences == 0:
            invalid_preferences_str = "No invalid preferences\n\n"
        else: 
            invalid_preferences_str = f"Invalid preferences: {invalid_preferences}\n" + invalid_preferences_str + "\n"
        if self.nplayers == 2:
            val_str = f"Value head: {value_head.item():.3f}\n"
        else:
            val_str = "Value head: "
            for i in range(self.nplayers):
                val_str += f"{value_head[i] * 100:.1f}%, "
            val_str = val_str[:-2] + "\n"
        card_str = get_card_str(info['cards'])
        game_str = f"Current player: NNPlayer (player {info['player']})\n" +\
                f"=======================================\n" + f"Obs: {obs}\n" + card_str + val_str\
                + invalid_preferences_str + "Action Probabilities:"
        print_evals_and_info(info, action_prob, game_str)

def ctv(card):
    card_to_value = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1]
    return card_to_value[card%13]

CARD_TO_VALUE_HAS_STARTER = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 15]
# mean_cards_in_hand = sum(DEAL_ORDER) / len(DEAL_ORDER)
MEAN_CARD_VALUE =  sum(CARD_TO_VALUE_HAS_STARTER) / 13
# MEAN_HAND_VALUE  = mean_cards_in_hand * mean_cards_value
START_PENALTY = 15
CARD_TO_VALUE = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1]
MEAN_CARD_DISTANCE = np.mean(CARD_TO_VALUE)
MAX_NORMALIZATION = 2
MIN_NORMALIZATION = 0.25

def symmetric_eval_board(obs, info):
    nplayers = len(obs)
    current_player = info['player']
    board_len = nplayers * PLACES_PER_SEGMENT
    base_bonus = 2 * board_len
    useable_fours = 0
    current_start_pawns = 0
    scores = np.zeros(nplayers)

    cards = info['cards']
    ncards = len(cards)
    cards = [cards[i] % 13 for i in range(ncards)]
    best_card_value = max([CARD_TO_VALUE[card] for card in cards])

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
            else:
                scores[player] += loc
                if player == current_player:
                    if loc <= 4:
                        useable_fours += 1
                    if loc > board_len - best_card_value:
                        scores[player] += loc
                        continue
                if loc > board_len - 2 * MEAN_CARD_DISTANCE:
                    scores[player] += .5 * loc

    # relative card value in player's hand
    card_score = 0
    # undid_start_penalty = 0
    # used_fours = 0
    for card in cards:
        if current_start_pawns and card in [11, 12]:
            useable_fours += 1
            current_start_pawns -= 1
            card_score += START_PENALTY
            # undid_start_penalty += 1
        if card == 2 and useable_fours:
            card_score += 1.5 * board_len
            useable_fours -= 1
            # used_fours += 1
        else:
            card_score += CARD_TO_VALUE[card]

    opp_card_score = MEAN_CARD_VALUE * (ncards - 0.5)
    if info['action'] == None and DEAL_ORDER[info['round']%3 - 1] != ncards:
        opp_card_score *= (nplayers - 2) / (nplayers - 1)
    relative_card_score = card_score - opp_card_score
    # add relative MEAN_CARD_VALUE to designate move advantage
    scores[current_player] += relative_card_score

    # print(f"Scores:")
    # for player, score in enumerate(scores):
    #     print(f"Player {player}: {score:.0f}")
    # if used_fours > 0:
    #     breakpoint()
    # if undid_start_penalty > 2:
    #     breakpoint()

    # rotate scores to player's perspective and normalize
    scores_from_player_perspective = np.array([scores[(i + current_player) % nplayers] for i in range(nplayers)])

    # normalize scores. If endgame, probabilities become sharper.
    max_score = np.max(scores_from_player_perspective)
    normalization_modifier = MAX_NORMALIZATION + (MIN_NORMALIZATION - MAX_NORMALIZATION) * max_score / (base_bonus * 4 + 25)
    # print(f"Normalization modifier: {normalization_modifier:.2f}")
    normalization_factor = normalization_modifier * board_len
    normalized_scores = scores_from_player_perspective / normalization_factor
    if np.max(normalized_scores) > 90:
        print(f"Error: max = {np.max(normalized_scores)}")
        breakpoint()
    winning_chances = np.exp(normalized_scores) / np.sum(np.exp(normalized_scores))
        
    return winning_chances

EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

class TreeSearchPlayer:
    def __init__(self, *, maxdepth=3, evalfunc=symmetric_eval_board, maxit=50, tau = .15, return_type="best_action", store_rng=True, alpha_beta_pruning = True, cache_enabled = True, verbose=False):
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
        self.lookups += 1
        entry = None
        if self.cache_enabled:
            game_hash = game.get_hash()
            if (game_hash, depth) in self.cache:
                self.hits += 1
                if self.alpha_beta_pruning:
                    entry = self.cache[(game_hash, depth)]
                    if entry['depth'] > depth:
                        entry = None
                    elif entry['type'] == EXACT:
                        return entry['value']
                    elif entry['type'] == LOWER_BOUND:
                        alpha = max(entry['value'][0], alpha)
                    elif entry['type'] == UPPER_BOUND:
                        beta = min(entry['value'][0], beta)
                else:
                    return self.cache[(game_hash, depth)]

        obs, rew, done, info = game._get_gamestate()
        current_player = info['player']
        nplayer = len(obs)
        alpha_orig = alpha
        if info['winner'] != -1:
            vals = np.zeros(nplayer)
            vals[(info['winner'] - current_player) % nplayer] = 1
            return vals
        if depth >= self.maxdepth:
            return self.eval(obs, info)
        nactions = len(info['space'])
        win_ratios = np.zeros((nactions, nplayer))
        win_ratios[:,1] = 1
        for i, action in enumerate(info['space']):
            newobs, rew, done, newinfo = game.step(action)
            new_current_player = newinfo['player']
            # print(f"Now at child {i}, (alpha, beta) = ({alpha}, {beta})")
            if new_current_player == current_player:
                win_ratio = self.tree_search(game, depth+1, alpha, beta)
                game.unmove()
            else:
                if nplayer == 2:
                    win_ratio = self.tree_search(game, depth+1, 1 - beta, 1 - alpha)
                else:
                    win_ratio = self.tree_search(game, depth+1, 0, 1 - alpha)

                game.unmove()
            player_diff = current_player - new_current_player
            for j, val in enumerate(win_ratio):
                win_ratios[i][(j - player_diff) % nplayer] = val
            if win_ratios[i][0] > alpha:
                # print(f"Setting alpha: {alpha}->{win_ratios[i][0]}")
                alpha = win_ratios[i][0]
            if self.alpha_beta_pruning:
                if alpha >= beta:
                    # print(f"alpha: {alpha}, beta: {beta}")
                    # print("Pruning tree!")
                    return win_ratios[np.argmax(win_ratios[:, 0])]

        best_win_ratio = win_ratios[np.argmax(win_ratios[:, 0])]
        if self.cache_enabled:
            if self.alpha_beta_pruning:
                if entry == None:
                    # No entry exists yet / there is only a shallow entry
                    new_entry = {
                            'value': best_win_ratio,
                            'depth': depth,
                    }
                    if best_win_ratio[0] <= alpha_orig:
                        new_entry['type'] = UPPER_BOUND
                    elif best_win_ratio[0] >= beta:
                        new_entry['type'] = LOWER_BOUND
                    else:
                        new_entry['type'] = EXACT
                    self.cache[(game_hash, depth)] = new_entry
            else:
                self.cache[(game_hash, depth)] = best_win_ratio
        if depth == 0:
            return win_ratios
        return best_win_ratio

    def show_eval_results(self, game, win_ratios):
        obs, rew, done, info = game._get_gamestate()
        player_str = f"Current player: TreeSearchPlayer, eval={self.eval.__name__} (player {info['player']})\n" + game.__repr__()
        game_str = f"\nLookups: {self.lookups}\nCache hit ratio: {self.hits / self.lookups * 100:.1f}%\n"
        game_str += player_str + "\nWinning chance per move: "
        print_evals_and_info(info, win_ratios[:,0], game_str)
    
    def move(self, game):
        self.hits = 0
        self.lookups = 0
        self.cache = {}
        obs, rew, done, info = game._get_gamestate()
        if self.store_rng:
            fastgame = FastTockGame(len(obs), max_his_len=self.maxdepth+1, store_rng_his=True)
        else:
            fastgame = FastTockGame(len(obs), max_his_len=self.maxdepth+1, store_rng_his=False)
        fastgame.load_full(game)
        win_ratios = self.tree_search(fastgame, 0, -1, 2)
        if self.verbose:
            self.show_eval_results(game, win_ratios)
        if self.return_type == "evals":
            return win_ratios
        elif self.return_type == "probabilities":
            scaled_ratios = np.exp(win_ratios[:, 0] / self.tau)
            prob = scaled_ratios / np.sum(scaled_ratios)
            return prob
        obs, rew, done, info = game._get_gamestate()
        best_action = info['space'][np.argmax(win_ratios[:, 0])]
        return best_action
    
    def determinization_move(self, obs, info):
        self.hits = 0
        self.lookups = 0
        self.cache = {}
        fastgame = FastTockGame(nplayers=len(obs), max_his_len=self.maxdepth+1)
        fastgame.load_game(obs, info)
        win_ratios = np.zeros((len(info['space']), len(obs)))
        # print(f"Actual action space is {info['space']}")
        for i in range(self.maxit):
            # print(f"Entering iteration {i} with state:")
            # print(fastgame)
            win_ratios += self.tree_search(fastgame, 0, -1, 2)
            fastgame._shuffle()
            fastgame.deal()
            fastgame.load_cards(info['player'], info['cards'][:])
        win_ratios /= self.maxit
        if self.verbose:
            self.show_eval_results(fastgame, win_ratios)
        if self.return_type == "evals":
            return win_ratios
        elif self.return_type == "probabilities":
            scaled_ratios = np.exp(win_ratios[:, 0] / self.tau)
            prob = scaled_ratios / np.sum(scaled_ratios)
            return prob
        best_action = info['space'][np.argmax(win_ratios[:, 0])]
        return best_action

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
        val = ctv(info['cards'][card])
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
        val = ctv(info['cards'][card])
        rel_loc = obs[player][pawn]
        capture_score = -100
        if rel_loc != 0:
            # capture an opponent's pawn
            loc = _abs_loc(rel_loc, player)
            if val == 0:
                continue
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
        val = ctv(info['cards'][card])
        rel_loc = obs[player][pawn]
        if val + rel_loc > board_len:
            return action

    # moveback action
    for action in info['space']:
        card, pawn = action
        rel_loc = obs[player][pawn]
        val = ctv(info['cards'][card])
        if val == -4:
            if rel_loc in [1, 2, 3, 4]:
                return action

    # capture advanced piece
    capture_action = get_capture(obs, info)
    if capture_action:
        return capture_action
    
    eager_action = eager_player(obs, info)
    card, pawn = eager_action
    val = ctv(info['cards'][card])
    if obs[player][pawn] != 0 and val in [1, 2, 3, -4]:
        # move out pawn
        for action in info['space']:
            card, pawn = action
            rel_loc = obs[player][pawn]
            if rel_loc == 0:
                return action
    return eager_action

def main():
    nplayers = 4
    game = make(nplayers)
    ts_player = TreeSearchPlayer(verbose=True, return_type="probabilities", alpha_beta_pruning = True, maxdepth=5)
    mcts_player = NNPlayer(fname = "weights/" + ("two" if nplayers == 2 else "four") + "_player.pth", verbose=True, return_type="eval", nplayers = nplayers, hyperparams={"maxit":1, "verbose": True})
    movefunc = mcts_player.mcts_move
    startmove = 20
    while True:
        move_no = 0
        obs, rew, done, info = game.reset()
        while not done:
            move_no += 1
            print(f"\n---------------------------------")
            print(f"Round {info['round']}")
            print(f"Player {info['player']} to move")
            print(game)
            if move_no > startmove:
                res = movefunc(obs, info)
                cont = input("------ Press Enter To Continue -------")

            move = choice(info['space'])
            print(f"Chose random move: {move}")
            obs, rew, done, info = game.step(move)
        print("Game finished")
        print(f"Player {info['winner']} won")

PLAYER_MAP  = {"TSPEnv": (TreeSearchPlayer, "move"),
               "TSPDet": (TreeSearchPlayer, "determinization_move"),
               "NN": (NNPlayer, "move"),
               "Smart": (None, smart_player),
               "Eager": (None, eager_player),
               "Random": (None, random_player),
               "NNMCTS": (NNPlayer, "mcts_move"),
               "VNNMCTS": (NNPlayer, "virt_mcts_move"),
               }

if __name__ == '__main__':
    main()
