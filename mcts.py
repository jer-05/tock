import random
from random import choice
import numpy as np
import copy
import torch
import traceback
import graphviz
import time

from tock import make, PLACES_PER_SEGMENT, DEAL_ORDER, RANKS
from fasttock import FastTockGame, ACTION_TUPLES
from network import PolicyNN
from utils import getstate, totorch, get_action_prob, revert_action_prob, get_action_string, print_evals_and_info, get_card_str, get_obs_str, get_unique_actions

CURRENT_PLAYER = 0

class Node:
    def __init__(self, parent, nplayers):
        self.parent = parent
        self.n = 0
        if nplayers == 2:
            self.s = 0
        else:
            self.s = np.zeros(nplayers)
        self.children = None
        self.player_dist = np.zeros(nplayers)
        self.nplayers = nplayers

    def generate_children(self):
        self.children = [Node(self, self.nplayers) for i in range(52)]

    def get_child_s_n(self, mask):
        try:
            child_s = np.array([self.children[i].s for i in range(52)])[mask]
            child_n = np.array([self.children[i].n for i in range(52)])[mask]
        except:
            traceback.print_exc()
            breakpoint()
        return child_s, child_n

    def __repr__(self):
        s  =  "------------ Node Info -------------\n"
        if self.parent:
            i = self.parent.children.index(self)
            card = i // 4
            pawn = i % 4
            s += f"Action: ({RANKS[card]}, {pawn})\n"
        else:
            s += f"Node is root node\n"
        s += f"N: {self.n}\n"
        if self.nplayers == 2:
            s += f"S: {self.s:.3f}\n"
        else:
            sstr = ""
            for i in range(self.nplayers):
                sstr += f"{self.s[i]:.3f}"
                if i != self.nplayers -1:
                    sstr += ", "
            s += f"S: {sstr}\n" 
        if self.children:
            s += f"Node has children\n"
        else:
            s += f"Node is a leaf node\n"
        s += "--------------- * -------------------\n"
        return s

    def print_children_and_self(self):
        print(self)
        for i, child in enumerate(self.children):
                if child.n > 0:
                    print(child)

    def print_hot_path(self,*,  is_leader_flag=True):
        if is_leader_flag:
            print("\n===========================")
            print("Showing hot path from node:")
        print(self)
        if self.children is None:
            print("Best variation ended here")
            print("=============================\n")
            return
        best_child = None
        highest_n = -1
        for i, child in enumerate(self.children):
            if child.n > highest_n:
                best_child = child
                highest_n = child.n
        best_child.print_hot_path(is_leader_flag=False)

def print_tree(game, root, root_action_prob, root_value, puct_scores, maxdepth=3, maxit="?", *, max_branching = 4, root_player):
    dot = graphviz.Digraph()
    nplayers = root.nplayers

    label = f"MCTS nodes after {maxit} iterations (N:= Visits, V:= Win Rate)"
    dot.attr(label=label, labelloc='t', fontsize='20', fontname="Helvetica-bold")

    dot.attr('node', shape='box', style='filled', fontname="Helvetica")

    root_n = root.n
    def _get_label(node, get_player_dist=True):
        dist = node.player_dist
        string = ""
        string += f"N: {node.n}\n"
        if nplayers == 2:
            string += f"V: {node.s/node.n:.2f}\n"
        else:
            string += "V: "
            for i in range(nplayers):
                string += f"{node.s[i]/node.n:.2f}, "
            string = string[:-2] + "\n"

        if get_player_dist:
            for i in range(len(dist)):
                string += f"P{i}: {dist[i]/node.n*100:.1f}%\n"
        return string

    def _get_colors(child):
        if nplayers == 2:
            value = int(child.s/child.n * 255)
        else:
            childs = (child.s[root_player] - .5) * 2
            value = int(childs/child.n * 255)
        r = min(255, 255 + value)
        g = 255 - abs(value)
        b = min(255, 255 - value)
        value_hexcode = f"#{r:02x}{g:02x}{b:02x}"
        return value_hexcode

    def _n_children(node):
        if node.children is None:
            return 0, 0
        child_n = np.array([child.n for child in node.children])
        n_children = np.sum(child_n != 0)
        return n_children, child_n

    def _add_node(node, depth):
        node_id = str(id(node))

        if depth > maxdepth or node.children is None:
            return 

        n_children, child_n = _n_children(node)

        if node != root:
            best_child_idx = np.argsort(child_n)[::-1]
            best_children = [node.children[best_child_idx[i]] for i in range(max_branching)]
        else:
            best_children = node.children
            best_child_idx = range(len(node.children))
        s = 0
        for idx, child in enumerate(best_children):
            if child.n > 0:
                i = best_child_idx[idx]
                child_id = str(id(child))
                child_label = _get_label(child)
                action_label = f"({RANKS[i // 4]}, {i %4})\n"
                if node == root:
                    action_tuple = (shifted_cards.index(i // 4), i % 4)
                    action_idx = root_actions.index(action_tuple)
                    Q, U = root_q[s], root_u[s]
                    if nplayers != 2:
                        Q = Q[root_player]
                        U = U.item()
                    try: 
                        action_label += f"Policy={root_action_prob[s]*100:.1f}%\nQ={Q:.3f}\nU={U:.3f}\n"
                    except:
                        traceback.print_exc()
                        breakpoint()
                    obs, rew, done, info = game.step(action_tuple)
                    obs_str = get_obs_str(obs, 0)
                    child_label += obs_str
                    game.unmove()
                val_hex = _get_colors(child)
                n_children, child_n = _n_children(child)
                desc = ""
                if n_children == 0:
                    desc = "LEAF"
                elif depth <= maxdepth:
                    missing = max(0, n_children - max_branching)
                    if missing:
                        desc = f"+{missing}"
                else:
                    missing = n_children
                    desc = f"+{missing}"
                dot.node(child_id, label=child_label, fillcolor=val_hex, xlabel=desc)
                visit = int(child.n/node.n * 200)
                visit_hexcode = f"#{200-visit:02x}ff{200-visit:02x}"
                visit_width = child.n / root_n * 100
                width = child.n / root_n * 5
                dot.edge(node_id, child_id, label=action_label, color=visit_hexcode, penwidth=f"{2 + width}")
                _add_node(child, depth+1)
                s += 1

    obs, rew, done, info = game._get_gamestate()
    root_q, root_u = puct_scores
    shifted_cards = [card % 13 for card in info['cards']]
    card_str = get_card_str(info['cards'])
    obs_str = get_obs_str(obs, 0)
    root_player = info['player']
    root_actions = info['space']
    root_id = str(id(root))
    root_label = _get_label(root, get_player_dist=False)
    if nplayers == 2:
        val_str = f"Value Head: {root_value:.3f}\n"
    else:
        val_str = f"Value Head: "
        for i in range(nplayers):
            val_str += f"{root_value[i]:.3f}, "
        val_str = val_str[:-2] + "\n"
    root_label += f"Player {root_player}\n" + val_str + card_str + obs_str
    val_hex = _get_colors(root)
    dot.node(root_id, root_label, fillcolor=val_hex)
    _add_node(root, 0)
    return dot

class MCTS:
    def __init__(self, obs, info, evalNN=None, *, hyperparams={}, verbose=False):
        self.nplayers = len(obs)
        self.root = Node(None, self.nplayers)
        self.root.generate_children()
        self.player = info['player']
        self.cards = info['cards'][:]
        if verbose:
            print(f"Loading MCTS with game:")
            print(f"Obs: {obs}")
            print(f"Info: {info}")
            print(f"Set self.player to {self.player} ")
            print(f"Set self.cards to {self.cards} ({get_card_str(self.cards)})")
        self.game = FastTockGame(len(obs))
        self.game.load_game(obs, info)
        self.info = copy.deepcopy(info)
        self.eval = evalNN
        self.cache = {}
        self.verbose = verbose
        self.hyperparams = {
                "epsilon": 0.25,
                "alpha": .3,
                "cpuct": 2,
                "maxit": 50,
        }
        for key, value in hyperparams.items():
            self.hyperparams[key] = value
        self.add_to_cache(obs, info, self.game.get_player_hash())


    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def get_child_indices(self, actionspace, cards):
        return [(cards[card_idx] % 13) * 4+ pawn for (card_idx, pawn) in actionspace]

    def get_puct_scores(self, child_s, child_n, parent_n, policy_heads, add_noise=False):
        nactions = len(policy_heads)
        eps = self.hyperparams['epsilon']
        alpha = self.hyperparams['alpha']
        cpuct = self.hyperparams['cpuct']
        if add_noise:
            dirichlett_noise = np.random.dirichlet([alpha] * nactions)
            policy = (1 - eps) * policy_heads + eps * dirichlett_noise
        else:
            policy = policy_heads
        q_value = child_s / (child_n + 1e-8)
        if self.nplayers == 2:
            u_value = cpuct * policy * np.sqrt(parent_n) / (child_n + 1)
        else:
            u_value = (cpuct * policy * np.sqrt(parent_n) / (child_n.reshape(-1) + 1)).reshape(-1, 1)

        return q_value, u_value

    def add_to_cache(self, obs, info, this_hash):
        state = getstate(obs, info, gettorch=True)
        self.eval.eval()
        with torch.no_grad():
            logits, value_head = self.eval(*state)
            policy_head = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            if self.nplayers == 2:
                value_head = value_head.item()
            else:
                value_head = torch.softmax(value_head, dim=1)[0].detach().cpu().numpy()

        self.cache[this_hash] = policy_head, value_head
        return policy_head, value_head


    def traverse(self, node, action_space, cards):
        # self._print(f"\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        # self._print(f"Traversing tree, now at node:\n{node}")
        # self._print(self.game)
        # self._print(f"Action space is {[get_action_string(action, cards) for action in action_space]}")
        # self._print(f"{get_card_str(cards)}")
        # compute policy heads for all children in advance (because batch advantage)
        unique_action_space = get_unique_actions(action_space, cards)
        # self._print(f"Unique space is {[get_action_string(u_act, cards) for u_act in unique_action_space]}")
        # if not self.use_gpu_manager:
        #     value_heads, child_indices = self.compute_child_policy_value(node, unique_action_space, cards)
        # else:
        #     child_indices = self.get_child_indices(action_space, cards)
        #     child_indices.sort()
        child_indices = self.get_child_indices(action_space, cards)
        child_indices.sort()

        # Traverse the tree
        this_hash = self.game.get_player_hash()
        if this_hash not in self.cache:
            obs, rew, done, info = self.game._get_gamestate()
            current_policy, current_value = self.add_to_cache(obs, info, this_hash)
        else:
            current_policy, current_value = self.cache[this_hash]

        action_prob, action_mask, idx_order = get_action_prob(current_policy, {'space': unique_action_space, 'cards': cards}, return_mask = True)

        if self.nplayers == 2:
            Q, U = self.get_puct_scores(*node.get_child_s_n(action_mask), node.n, action_prob[idx_order], add_noise=(node.parent == None))
        else:
            s, n = node.get_child_s_n(action_mask)
            Q, U = self.get_puct_scores(s, n.reshape(-1, 1), node.n, action_prob[idx_order], add_noise=(node.parent == None))

        current_player = self.game.state[CURRENT_PLAYER]
        if self.nplayers == 2:
                puct_scores = self.get_relative_value(Q, current_player) + U
                rel_idx = np.argmax(puct_scores)
        else:
            puct_scores = Q + U
            rel_idx = np.argmax(puct_scores[:, current_player])
        # # self._print(f"Got puct scores {puct_scores} for node")
        # # self._print(node)
        try:
            best_child_idx = child_indices[rel_idx]
        except:
            traceback.print_exc()
            breakpoint()

        best_child = node.children[best_child_idx]
        pr_act = unique_action_space[np.argsort(idx_order)[rel_idx]]
        # self._print(f"Chose best child {rel_idx}:{pr_act}")
        # self._print(f"Corresponds to action {get_action_string(pr_act, cards)}")
        obs, rew, done, info = self.game.step(pr_act)
        best_child.player_dist[info['player']] += 1
        if done:
            if self.nplayers == 2:
                value = self.get_relative_value(1, info['winner'])
            else:
                value = np.array([1 if i == info['winner'] else 0 for i in range(self.nplayers)])
            # self._print(f"Child is terminal with value {value}")
            return best_child, value
        elif best_child.children == None:
            if best_child.n == 0:
                game_hash = self.game.get_player_hash()
                if game_hash not in self.cache:
                    policy, value = self.add_to_cache(obs, info, game_hash)
                else:
                    policy, value = self.cache[game_hash]
                if self.nplayers == 2:
                    value = self.get_relative_value(value, info['player'])
                else:
                    value = self.get_relative_value(value, info['player'])
                # self._print(f"Child is a leaf and has n=0")
                return best_child, value
            else:
                best_child.generate_children()
                return self.traverse(best_child, info['space'], info['cards'])
        return self.traverse(best_child, info['space'], info['cards'])

    def get_relative_value(self, value, player):
        if self.nplayers == 2:
            if player == self.player:
                return value
            else:
                return -value
        else:
            column_order = np.array([(i - player) % self.nplayers for i in range(self.nplayers)])
            if len(value.shape) == 1:
                return value[column_order]
            else:
                return value.T[column_order].T

    def backpropagate(self, node, value):
        # self._print("Backpropagating")
        while node != None:
            node.n += 1
            node.s += value
            node = node.parent
        self.game.rewind()

    def iterate(self):
        for i in range(self.hyperparams['maxit']):
            # self._print(f"Now doing iteration {i+1}")
            self.game._shuffle()
            self.game.deal(ncards=len(self.cards))
            self.game.load_cards(self.player, self.cards[:])
            # # self._print(self.game)
            obs, rew, done, info = self.game._get_gamestate()
            leaf, value = self.traverse(self.root, info['space'][:], info['cards'][:])
            # try:
            #     # self._print(f"Ended traversal at leaf (value={value:.3f}):\n{leaf}") 
            # except:
            #     print(f"Got value {value}, type {type(value)} (bad shape?)")
            #     exit(1)
            self.backpropagate(leaf, value)

    def get_results(self, verbose=False):
        obs, rew, done, info = self.game._get_gamestate()
        actual_space = info['space'][:]
        actual_cards = info['cards'][:]
        mask = np.ones(52).astype(bool)
        s, n = self.root.get_child_s_n(mask)
        if self.nplayers != 2:
            s = s[:,self.player]
        val = s / (n + 1e-8)
        prob = n / self.root.n
        unique_actions = get_unique_actions(info['space'], info['cards'])
        info['space'] = unique_actions
        action_prob = get_action_prob(prob, info)
        child_indices = self.get_child_indices(info['space'], info['cards'])
        child_values = [val[child_indices[i]] for i in range(len(child_indices))]
        if self.verbose or verbose:
            self.root.print_children_and_self()
            self.root.print_hot_path()
            print_evals_and_info(info, np.vstack((action_prob, child_values)), f"MCTS results (player {info['player']}):\n" + self.game.__repr__() + f"\nAction Probabilities and Values: ", twoD_eval=True)
        action_prob = get_action_prob(prob, info)
        # breakpoint()
        return action_prob, val

    def save_tree(self, fname, maxdepth=2, max_branching=3):
        obs, rew, done, info = self.game._get_gamestate()
        
        root_policy, root_val = self.cache[self.game.get_player_hash()]
        action_space = info['space']
        cards = info['cards']
        unique_action_space = get_unique_actions(action_space, cards)
        action_prob, action_mask, idx_order = get_action_prob(root_policy, {'space':unique_action_space, 'cards':cards}, return_mask = True)
        ordered_action_prob = action_prob[idx_order]
        if self.nplayers == 2:
            puct_scores = self.get_puct_scores(*self.root.get_child_s_n(action_mask), self.root.n, ordered_action_prob)
        else:
            s, n = self.root.get_child_s_n(action_mask)
            puct_scores = self.get_puct_scores(s, n.reshape(-1, 1), self.root.n, ordered_action_prob)

        dot = print_tree(self.game, self.root, ordered_action_prob, root_val, puct_scores, maxdepth, self.hyperparams['maxit'], max_branching=max_branching, root_player=self.player)
        dotfname = "temp.dot"
        dot.save(dotfname)
        dot.render(filename=dotfname, outfile=fname, cleanup=True)

def main():
    nplayers = 2
    game = make(nplayers)
    nw = PolicyNN(nplayers=nplayers)
    nw.load("weights/two_player.pth", silent=True, device='cpu')
    obs, done, rew, info = game._get_gamestate()
    hyperparams = {"maxit": 400, "cpuct": 2}
    move_no = 0
    while not done:
        try:
            if nplayers == 2 and move_no > 10 or (nplayers == 4 and move_no > 60):
                mcts = MCTS(obs, info, nw, hyperparams=hyperparams, verbose=True)
                mcts.iterate()
                mcts.save_tree(f"figures/mcts/mcts_tree_move-{move_no}.png")
                prob, val = mcts.get_results(verbose=True)
                move = info['space'][np.argmax(prob)]
                temp = input("--------- Press Enter to Continue ---------")
            # ts_player = TreeSearchPlayer(verbose=True, return_type="probabilities", alpha_beta_pruning = True, maxdepth=5)
            else:
                move = choice(info['space'])

            # print(f"prob: {prob}\nval: {val}")
            obs, rew, done, info= game.step(move)
            move_no += 1
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            breakpoint()

if __name__ == '__main__':
    main()

    
