import random
from random import choice
import numpy as np
import copy
import torch
import traceback
import time

from network import PolicyNN
from utils import getstate, totorch, get_action_prob, revert_action_prob, get_action_string, print_evals_and_info, get_card_str, get_obs_str, get_unique_actions
from mcts import Node, print_tree
from tock import make, PLACES_PER_SEGMENT, DEAL_ORDER, RANKS
from fasttock import FastTockGame, ACTION_TUPLES

CURRENT_PLAYER = 0
NPLAYERS = 2

class VMCTS:
    def __init__(self, obs, info, *, worker_id, gpu_manager_info, hyperparams={}, verbose=False):
        self.root = Node(None, len(obs))
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
        # force registration of history
        self.game.step(info['space'][0])
        self.game.unmove()
        # print(f"Loaded game state, action space is now:")
        # obs, rew, done, info = self.game._get_gamestate()
        # print(info['space'])
        self.nplayers = len(obs)
        self.info = copy.deepcopy(info)
        self.obs = copy.deepcopy(obs)
        self.worker_id = worker_id
        self.shared_obs = gpu_manager_info['shared_obs']
        self.shared_p = gpu_manager_info['shared_p']
        self.events = gpu_manager_info['events']
        self.shared_v = gpu_manager_info['shared_v']
        self.request_queue = gpu_manager_info['request_queue']
        self.cache = {}
        self.verbose = verbose
        self.hyperparams = {
                "epsilon": 0.25,
                "alpha": .3,
                "cpuct": 2,
                "maxit": 50,
                "num_sims": 16,
                "virtual_loss": 1,
        }
        for key, value in hyperparams.items():
            self.hyperparams[key] = value
        self.games = [copy.deepcopy(self.game) for i in range(self.hyperparams["num_sims"])]
        self.add_batch_to_cache([self.obs], [self.info], [self.game.get_player_hash()])

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
        u_value = cpuct * policy * np.sqrt(parent_n) / (child_n + 1)
        return q_value, u_value


    def get_relative_player(self, player):
        return (self.player - player) % self.nplayers

    def get_relative_value(self, value, player):
        if player == self.player:
            return value
        else:
            return -value

    def backpropagate(self, node, value):
        # print("Backpropagating")
        while node != None:
            node.n += 1
            node.s += value
            node = node.parent

    def traverse_generator(self, game):
        """A generator that yields whenever it needs an NN evaluation."""
        current_node = self.root
        cards = self.cards[:]
        player = self.player
        game.rewind()
        game._shuffle()
        game.deal(ncards=len(cards))
        game.load_cards(player, cards[:])
        # print(f"Firing up traverse generator with game:")
        # print(game)
        
        while True:
            # print(f"Now in generator {self.games.index(game)} (before yield)")
            obs, rew, done, info = game._get_gamestate()
            # print(game)
            # print(f"Obs: {obs}")
            # print(f"Info: {info}")
            if done:
                value = self.get_relative_value(1, info['winner'])
                # print(f"Child is terminal with value {value}")
                return current_node, value

            this_hash = game.get_player_hash()
            if this_hash not in self.cache:
                current_node.n += self.hyperparams['virtual_loss'] 
                # print("Yielding for evaluation")
                yield (copy.deepcopy(obs), copy.deepcopy(info), this_hash) 
                current_node.n -= self.hyperparams['virtual_loss']

            # print(f"Now in generator {self.games.index(game)} (after yield)")
            # print(game)
            # print(f"Obs: {obs}")
            # print(f"Info: {info}")

            policy, value = self.cache[this_hash]

            if current_node.children == None:
                if current_node.n == 0:
                    # print(f"Child is a leaf and has n=0")
                    return current_node, self.get_relative_value(value[0], info['player'])
                else:
                    current_node.generate_children()

            # print("Evaluating next step in traversal")

            action_space = info['space']
            cards = info['cards']
            child_indices = self.get_child_indices(action_space, cards)
            child_indices.sort()

            # Traverse the tree
            unique_action_space = get_unique_actions(action_space, cards)
            # this_hash = game.get_player_hash()
            current_policy, current_value = self.cache[this_hash]
            # print(f"Got policy {current_policy.reshape(13, 4)} and value {current_value} from cache")
            try:
                action_prob, action_mask, idx_order = get_action_prob(current_policy, {'space': unique_action_space, 'cards': cards}, return_mask = True)
            except:
                # print("Action prob error !!!")
                # print(f"policy: {current_policy}")
                # print(f"Space: {unique_action_space}")
                # print(f"Cards: {cards}")
                breakpoint()
            
            Q, U = self.get_puct_scores(*current_node.get_child_s_n(action_mask), current_node.n, action_prob[idx_order], add_noise=(current_node.parent == None))
            current_player = game.state[CURRENT_PLAYER]
            puct_scores = self.get_relative_value(Q, current_player) + U
            # print(f"Got puct scores {puct_scores} for node")
            # print(current_node)
            rel_idx = np.argmax(puct_scores)
            best_child_idx = child_indices[rel_idx]


            best_child = current_node.children[best_child_idx]
            pr_act = unique_action_space[np.argsort(idx_order)[rel_idx]]
            # print(f"Chose best child {rel_idx}:{pr_act}")
            # print(f"Corresponds to action {get_action_string(pr_act, cards)}")
            obs, rew, done, info = game.step(pr_act)
            best_child.player_dist[info['player']] += 1
            current_node = best_child

    def add_batch_to_cache(self, obss, infos, hashes):
        maxbatch = self.hyperparams['num_sims']
        nbatch = len(obss)
        # print(f"Adding {nbatch} batches to queue")
        startidx = self.worker_id * maxbatch
        endidx = startidx + nbatch
        states = [getstate(obss[i], infos[i], gettorch=True, unsqueeze=False) for i in range(nbatch)]
        transposed_states = [torch.stack(tensors) for tensors in zip(*states)]
        for i in range(len(transposed_states)):
            self.shared_obs[i][startidx:endidx].copy_(transposed_states[i]) # Fast memory copy
        self.request_queue.put((self.worker_id, nbatch))
        self.events[self.worker_id].wait()                # Wait for GPU
        self.events[self.worker_id].clear()
        policy_heads = self.shared_p[startidx:endidx].numpy()    # Result is already there!
        value_heads = self.shared_v[startidx:endidx].numpy()
        for i, hash_ in enumerate(hashes):
            self.cache[hash_] = (policy_heads[i].copy(), value_heads[i].copy())
        # print(f"Got results for {nbatch} batches")
        return policy_heads, value_heads

    def iterate(self):
        # print(f"Starting simultaneous simulations")
        searches = [self.traverse_generator(self.games[_]) for _ in range(self.hyperparams['num_sims'])]
        iteration = 0
        while True:
            # print(f"Now at iteration {iteration+1}")

            next_requests = []
            next_searches = []
            need_batch_eval = False
            for i, s in enumerate(searches):
                # print(f"Now advancing search {i}")
                try:
                    next_req = s.send(None)
                    next_searches.append(s)
                    next_requests.append(next_req)
                    need_batch_eval = True
                except StopIteration as e:
                    # print(f"Search {i} concluded")
                    endnode, val = e.value
                    self.backpropagate(endnode, val)
                    next_searches.append(self.traverse_generator(self.games[i]))
                    iteration += 1
            if iteration >= self.hyperparams['maxit']:
                return
            searches = next_searches
            if not need_batch_eval:
                continue
            active_requests = next_requests

            batch_obs = [req[0] for req in active_requests]
            batch_info = [req[1] for req in active_requests]
            batch_hashes = [req[2] for req in active_requests]

            policies, values = self.add_batch_to_cache(batch_obs, batch_info, batch_hashes)
        # print(f"Finished simulations!")

    def get_results(self, verbose=False):
        obs, rew, done, info = self.game._get_gamestate()
        actual_space = info['space'][:]
        actual_cards = info['cards'][:]
        mask = np.ones(52).astype(bool)
        s, n = self.root.get_child_s_n(mask)
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
        puct_scores = self.get_puct_scores(*self.root.get_child_s_n(action_mask), self.root.n, ordered_action_prob)
        dot = print_tree(self.game, self.root, ordered_action_prob, root_val, puct_scores, maxdepth, self.hyperparams['maxit'], max_branching=max_branching)
        dotfname = "temp.dot"
        dot.save(dotfname)
        dot.render(filename=fname, outfile=fname, cleanup=True)

def main():
    game = make(2)
    nw = PolicyNN()
    model_path = "weights/evo-0_cycle-0.pth"
    nw.load(model_path)
    obs, done, rew, info = game._get_gamestate()
    hyperparams = {"maxit": 800, "cpuct": 2}
    move_no = 0
    while not done:
        try:
            if move_no > 10:
                mcts = VMCTS(obs, info, 0, hyperparams=hyperparams, verbose=True)
                mcts.iterate()
                mcts.save_tree(f"figures/mcts_tree_move-{move_no}.png")
                prob, val = mcts.get_results(verbose=True)
                move = info['space'][np.argmax(prob)]
                temp = input("--------- Press Enter to Continue ---------")
            # ts_player = TreeSearchPlayer(verbose=True, return_type="probabilities", alpha_beta_pruning = True, maxdepth=5)
            else:
                move = choice(info['space'])

            print(f"prob: {prob}\nval: {val}")
            obs, rew, done, info= game.step(move)
            move_no += 1
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            breakpoint()

if __name__ == '__main__':
    main()

    
