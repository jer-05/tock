import random
from random import choice
import numpy as np
import copy
import torch
import torch._inductor.config as config
config.fx_graph_cache = True
import traceback
import graphviz
import time
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath("./compiled_model_cache")
import torch.onnx
import onnxruntime as ort
import cProfile



from tock import make, PLACES_PER_SEGMENT, DEAL_ORDER, RANKS
from fasttock import FastTockGame, ACTION_TUPLES, return_search_results, print_action_space, print_cards, CONCISE_ACTION_IDX_TO_NAME
from network import PolicyNN
from utils import getstate, totorch, get_action_string, get_card_str, get_obs_str, get_ort_inference_session, print_block, print_important, print_v

import cProfile

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
        assert self.children is None
        self.children = [Node(self, self.nplayers) for i in range(50)]

    def get_child_s_n(self, mask):
        try:
            child_s = np.array([self.children[i].s for i in range(50)])[mask]
            child_n = np.array([self.children[i].n for i in range(50)])[mask]
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

def print_tree(game, root, root_action_prob, root_value, root_action_mask, puct_scores, maxdepth=3, maxit="?", *, max_branching = 4, root_player, dirichlet_noise, Q, U):
    dot = graphviz.Digraph()
    nplayers = root.nplayers

    label = f"MCTS nodes after {maxit} iterations (N:= Visits, V:= Value)"
    dot.attr(label=label, labelloc='t', fontsize='20', fontname="Helvetica-bold")

    dot.attr('node', shape='box', style='filled', fontname="Helvetica")

    root_n = root.n
    def _get_label(node, get_player_dist=True):
        dist = node.player_dist
        string = ""
        string += f"N: {node.n}\n"
        if node.n > 0:
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
        if child.n == 0:
            value_hexcode = "#D3D3D3"
        else:
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

        if depth >= maxdepth or node.children is None:
            return 

        n_children, child_n = _n_children(node)

        if node != root:
            best_child_idx = np.argsort(child_n)[::-1]
            best_children = [node.children[best_child_idx[i]] for i in range(max_branching)]
        else:
            best_children = node.children
            best_child_idx = range(len(node.children))
        s = 0
        for action_number, child in enumerate(best_children):
            if child.n > 0 or node == root and root_action_mask[action_number]:
                i = best_child_idx[action_number]
                child_id = str(id(child))
                child_label = _get_label(child)
                action_label = f"{CONCISE_ACTION_IDX_TO_NAME[action_number]}\n"
                if node == root:
                    Q, U = root_q[s], root_u[s]
                    if nplayers != 2:
                        Q = Q[root_player]
                        U = U.item()
                    action_label += f"Policy={root_action_prob[s]*100:.1f}%\nDir. Noise={format(dirichlet_noise[s]*100, '.1f')+'%' if dirichlet_noise is not None else 'None'}\nQ={Q:.3f}\nU={U:.3f}\nPUCT={puct_scores[s]:.3f}"
                    obs, rew, done, info = game.step(action_number)
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
                if node.n != 0:
                    visit = int(child.n/node.n * 200)
                    width = child.n / root_n * 5
                else:
                    visit = 200
                    width = 5
                visit_hexcode = f"#{200-visit:02x}ff{200-visit:02x}"
                dot.edge(node_id, child_id, label=action_label, color=visit_hexcode, penwidth=f"{2 + width}")
                _add_node(child, depth+1)
                s += 1

    obs, rew, done, info = game._get_gamestate()
    root_q, root_u = Q, U
    shifted_cards = [card % 13 for card in info['cards']]
    card_str = get_card_str(info['cards'])
    obs_str = get_obs_str(obs, 0)
    root_player = info['player']
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
    def __init__(self, obs, info, return_type, inference_session, *, hyperparams={}, verbose=False, debug=False):
        self.debug=debug
        self.root_info = copy.deepcopy(info)
        self.root_obs = copy.deepcopy(obs)
        self.root_player = info['player']
        self.root_cards = info['cards'][:]
        self.nplayers = len(obs)
        self.root = Node(None, self.nplayers)
        self.root.generate_children()
        self.game = FastTockGame(len(obs), debug=self.debug)
        self.game.load_game(obs, info)
        self.inference_session = inference_session
        self.cache = {}
        self.lookups = 0
        self.cache_hits = 0
        self.verbose = verbose
        self.hyperparams = {}
        self.return_type = return_type
        if verbose:
            print(f"Initialized MCTS")
            print(f"Hyperparameters:")
        for key, value in hyperparams.items():
            self.hyperparams[key] = value
            if verbose:
                print(f"  {key}: {value}")
        obs, rew, done, info = self.game._get_gamestate()
        self.root_action_mask = info['space'][:]
        self.root_hash = self.game.get_NN_hash()
        self.root_value, self.root_action_prob = self.add_to_cache(obs, info, self.root_hash)
        self.only_move = False
        if len(self.root_action_prob) == 1:
            self.only_move = True
        if self.hyperparams["noise"]:
            self.dirichlet_noise = np.random.dirichlet([self.hyperparams["alpha"]] * int(np.sum(info['space'])))
            if verbose:
                print(f"Dirichlet noise: {', '.join([f'{noise*100:.1f}%' for noise in self.dirichlet_noise])}")
        else:
            self.dirichlet_noise = None
        if verbose:
            self.game.print_visible()
            print(f"Raw Network Policy and Value:")
            print(f"Value head: ",  f'{self.root_value:.3f}' if self.nplayers == 2 else ', '.join([format(v, ':3f') for v in self.root_value]))
            print_action_space(self.root_action_mask, action_probabilities=self.root_action_prob, check_normalization=True, print_delimiters=False)

    def get_puct_scores(self, child_s, child_n, parent_n, policy_heads, add_noise=False):
        """
        Return Q and U values that together make up the PUCT score for all
        child nodes
        """
        nactions = len(policy_heads)
        eps = self.hyperparams['epsilon']
        alpha = self.hyperparams['alpha']
        cpuct = self.hyperparams['cpuct']
        if add_noise and self.dirichlet_noise is not None:
            policy = (1 - eps) * policy_heads + eps * self.dirichlet_noise
        else:
            policy = policy_heads
        q_value = child_s / (child_n + 1e-8)
        if self.nplayers == 2:
            u_value = cpuct * policy * np.sqrt(parent_n) / (child_n + 1)
        else:
            u_value = (cpuct * policy * np.sqrt(parent_n) / (child_n.reshape(-1) + 1)).reshape(-1, 1)

        return q_value, u_value

    def add_to_cache(self, obs, info, this_hash):
        """
        Evaluate the neural network for a specific game state;
        store the result in cache to save on computation
        in case of later reference
        """
        self.lookups += 1
        cards = info['cards']
        state = getstate(obs, info, getdict=True)
        # compute the policy and value evaluations
        logits, value_head = self.inference_session.run(None, state)
        logits = logits[0]
        policy_head = np.exp(logits) / np.sum(np.exp(logits))
        if self.nplayers == 2:
            value_head = value_head.item()
        else:
            value_head = value_head[0]
            value_head = np.exp(value_head) / np.sum(np.exp(value_head))
        # Convert the 52 x 1 policy head to an array with action probabilities corresponding to the action space by masking and renormalizing
        try:
            self.cache[this_hash] = value_head, policy_head[state['action_mask'][0].astype(bool)]
        except Exception:
            traceback.print_exc()
            breakpoint()
        return value_head, policy_head[state['action_mask'][0].astype(bool)]

    def traverse(self, node, action_mask, cards):
        """
        Recursively traverse the tree of nodes from previous iterations
        until ending up at a leave node to return its value
        """
        if self.debug:
            print(f"Traversing tree from position:")
            print(self.game)
        try:
            # get action space and list of indices of valid children from the current game state. Children that cannot be reached in one move are ignored.
            child_indices = np.arange(50)[action_mask]
            obs, rew, done, info = self.game._get_gamestate()

            # get policy head (retrieve from cache if available)
            this_hash = self.game.get_NN_hash()
            if this_hash not in self.cache:
                obs, rew, done, info = self.game._get_gamestate()
                _, action_prob = self.add_to_cache(obs, info, this_hash)
            else:
                _, action_prob = self.cache[this_hash]
                self.cache_hits += 1

            # get PUCT scores of reachable child nodes
            current_player = self.game.state[CURRENT_PLAYER]
            if self.nplayers == 2:
                Q, U = self.get_puct_scores(*node.get_child_s_n(action_mask), node.n, action_prob, add_noise=(node.parent == None))
                puct_scores = self.get_relative_value(Q, current_player) + U
            else:
                s, n = node.get_child_s_n(action_mask)
                Q, U = self.get_puct_scores(s, n.reshape(-1, 1), node.n, action_prob, add_noise=(node.parent == None))
                puct_scores = Q[:, current_player] + U

            # choose best child
            best_idx = np.argmax(puct_scores)
            best_action_number = child_indices[best_idx]
            best_child = node.children[best_action_number]
            obs, rew, done, info = self.game.step(best_action_number)
            best_child.player_dist[info['player']] += 1

            # check if node is a leaf node
            if done:
                if self.nplayers == 2:
                    value = self.get_relative_value(1, info['winner'])
                else:
                    value = np.array([1 if i == info['winner'] else 0 for i in range(self.nplayers)])
                return best_child, value
            elif best_child.children == None:
                best_child.generate_children()
                child_game_hash = self.game.get_NN_hash()
                if child_game_hash not in self.cache:
                    value, _ = self.add_to_cache(obs, info, child_game_hash)
                else:
                    value, _ = self.cache[child_game_hash]
                    self.cache_hits += 1
                value = self.get_relative_value(value, info['player'])
                return best_child, value

            # node is not a leaf node; continue traversing
            return self.traverse(best_child, info['space'], info['cards'])
        except Exception as e:
            print(f"Error: {e}")
            self.game.print_visible()
            print_action_space(action_mask)
            print_cards(cards)
            print(flush=True)
            traceback.print_exc()
            breakpoint()

    def get_relative_value(self, value, player):
        if self.nplayers == 2:
            if player == self.root_player:
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
        """
        Update all nodes in the search path with the value
        obtained at the leaf node by retracing the leaf node's
        parent chain
        """
        if self.debug:
            val_str = f"{value:.3f}" if self.nplayers == 2 else ", ".join([f'{value[i]:.3f}' for i in range(len(value))])
            print_important(f"Backpropagating with value {val_str}")
        while node != None:
            node.n += 1
            node.s += value
            node = node.parent

    def iterate(self):
        """
        Execute maxit iterations of MCTS
        """
        if self.only_move:
            if self.verbose:
                print(f"[!]: MCTS.iterate: not iterating because the position admits only one legal move")
            return
        with torch.no_grad():
            for i in range(self.hyperparams['maxit']):
                if self.debug:
                    print_block(f"Starting iteration {i}")
                self.game._shuffle()
                self.game.deal(ncards=len(self.root_cards))
                self.game.load_cards(self.root_player, self.root_cards[:])
                leaf, value = self.traverse(self.root, self.root_action_mask[:], self.root_cards[:])
                if self.debug:
                    print(f"Ended traversal at node:")
                    print(leaf)
                self.backpropagate(leaf, value)
                self.game.rewind()

    def get_results(self):
        """
        Return the action probabilities and move valuations
        Action probabilities are proportional visit counts and depend
        only indirectly on move valuations
        """
        if self.verbose:
            print(f"Returning MCTS results:")
            print(f"    Evaluation lookups: {self.lookups}")
            print(f"    Cache hit ratio: {self.cache_hits / (self.lookups + self.cache_hits) * 100:.1f}%")
        if self.only_move:
            action_probabilities = np.array([1])
            action_values = np.array([self.root_value])
        else:
            s, n = self.root.get_child_s_n(self.root_action_mask)
            if self.nplayers != 2:
                s = s[:,self.root_player]
            action_values = s / (n + 1e-8)
            action_probabilities = n / self.root.n
        child_indices = np.arange(50)[self.root_action_mask]

        player_obs = self.root_obs[self.root_player]
        return return_search_results(self.return_type, self.root_action_mask, player_obs, self.root_info, action_probabilities, verbose=self.verbose, check_normalization=True, tau=None, action_values=action_values)

    def save_tree(self, fname, maxdepth=3, max_branching=2):
        obs, rew, done, info = self.game._get_gamestate()
        current_player = self.game.state[CURRENT_PLAYER]
        if self.nplayers == 2:
            Q, U = self.get_puct_scores(*self.root.get_child_s_n(self.root_action_mask), self.root.n, self.root_action_prob, add_noise=True)
            puct_scores = self.get_relative_value(Q, current_player) + U
        else:
            s, n = self.root.get_child_s_n(self.root_action_mask)
            Q, U = self.get_puct_scores(s, n.reshape(-1, 1), self.root.n, self.root_action_prob, add_noise=True)
            puct_scores = Q[:, current_player] + U
            Q = Q[:, current_player]
        dot = print_tree(self.game, self.root, self.root_action_prob, self.root_value, self.root_action_mask, puct_scores, maxdepth, self.hyperparams['maxit'], max_branching=max_branching, root_player=self.root_player, dirichlet_noise=self.dirichlet_noise, Q=Q, U=U)
        dotfname = "temp.dot"
        dot.save(dotfname)
        dot.render(filename=dotfname, outfile=fname, cleanup=True)

def main(ort_info):
    hyperparams = {"maxit": 1000, "cpuct": 2, "alpha": .2, "epsilon": .25, "noise": False}
    nplayers = 2
    game = make(nplayers)
    nw = PolicyNN(nplayers=nplayers)
    # fname = "weights/2_players/session39best.pth"

    fname = "weights/2_players/datasize_300005_epochs_10_batch_256_lr_0.001_iteration_1.pth"
    # fname = "weights/session35best.pth"
    nw.load(fname, silent=True)
    sess = get_ort_inference_session(nw, fname, 0, *ort_info)
    while True:
        move_no = 0
        obs, rew, done, info = game.reset()
        while not done:
            try:
                if nplayers == 2 and move_no >= 0 or (nplayers == 4 and move_no > 60):
                    mcts = MCTS(obs, info, "best_action", sess, hyperparams=hyperparams, verbose=True, debug=False)
                    mcts.iterate()
                    # mcts.save_tree(f"figures/mcts/mcts_tree.png")
                    # mcts.save_tree(f"figures/mcts/mcts_tree_move_{move_no}.png")
                    move = mcts.get_results()
                    assert move in info['space']
                    try:
                        temp = input("--------- Press Enter to Continue ---------")
                    except EOFError:
                        print(f"\nInterrupting game")
                        break

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
  from mp_ort_import import exec_main
  exec_main(main)

    
