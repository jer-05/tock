import torch
import torch.multiprocessing as mp
import numpy as np
import os
import multiprocessing
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time, sleep
import inspect
from tqdm import tqdm

from tock import make, AVERAGE_GAME_LENGTH
from fasttock import FastTockGame
from gpu_manager import get_gpu_manager
from players import get_players, get_fname, get_fullnames, clip_cfg
from network import PolicyNN
from utils import getstate, revert_action_prob, checkfn, remove_dirs, clear_lines

COMPETITION_FIG_DIR = "figures/competition"

def _play_n_games(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    player_names, player_configs, ngames, pid, results_queue = args
    e_threads = list(range(4, 32))
    os.sched_setaffinity(0, set(e_threads))
    for cfg in player_configs:
        if 'gpu_manager_info' in cfg:
            cfg['worker_id'] = pid
    players, types = get_players(player_names, player_configs)
    nplayers = len(player_names)
    env = make(nplayers)
    wins = np.zeros(len(players))
    timers = np.zeros(len(players))
    checkpoint = max(ngames // 10, 1)
    moves = 0
    start_wait = time()
    for game in range(ngames):
        obs, rew, done, info = env.reset()
        while not done:
            player_idx = (info['player'] + game) % nplayers
            player = players[player_idx]
            start = time()
            if types[player_idx] == 'environment':
                action = player(env)
            elif types[player_idx] == 'obsinfo':
                action = player(obs, info)
            timers[player_idx] += time() - start
            obs, rew, done, info = env.step(action=action)
            moves += 1
        wins[(info['winner'] + game) % nplayers] += 1
        if sum(wins) > 0 and time() - start_wait > .1:
            results_queue.put((wins, timers/moves, False))
            wins = np.zeros(len(players))
            timers = np.zeros(len(players))
            start_wait = time()
    if results_queue is None:
        return wins, timers / moves
    results_queue.put((wins, timers/moves, True))


class Competition:
    def __init__(self, player_names, player_configs, gamemode = 2, nthreads=os.cpu_count() - 1, gpu_managers=[]):
        self.gamemode = gamemode
        self.nthreads = nthreads
        self.gpu_managers = gpu_managers
        self.results_queue = mp.Queue()
        self.finish_events = [mp.Event() for i in range(self.nthreads)]
        assert (self.gamemode in [2, 4, 6]), f"Invalid game mode '{self.gamemode}'"
        assert (len(player_names) == len(player_configs)), f"Missing or redundant player configurations"
        assert (len(player_names) != 1), f"Only one contestant"
        self.player_names = player_names
        self.nplayers = len(self.player_names)
        if self.gamemode in [4, 6]:
            assert (self.nplayers == self.gamemode), f"Number of players and game mode does not match (multiple rounds for game mode {self.gamemode} players not implemented)"
        self.player_configs = player_configs
        self.clipped_configs = [clip_cfg(cfg) for cfg in player_configs]
   
    def play_round(self, player_names, player_configs, ngames, *,show_info):
        if show_info:
            print("Contestants are: ")
            for i, (name, cfg) in enumerate(zip(player_names, player_configs)):
                print(f"Player {i+1}: {name}")
                print(f"    has config {clip_cfg(cfg)}")
        wins = np.zeros(self.gamemode)
        timers = np.zeros(self.gamemode)
        games_per_core = max(1, ngames // self.nthreads)
        leftovers = ngames - self.nthreads * games_per_core
        if games_per_core == 1:
            leftovers = 0
            ngames = self.nthreads
        args = [(player_names, player_configs, games_per_core if (_!=0) else games_per_core + leftovers, _, self.results_queue) for _ in range(self.nthreads)]
        threads_finished = 0
        for gpu_mgr in self.gpu_managers:
            gpu_mgr.start()
        processes = []
        print(f"Launching {self.nthreads} workers")
        for i in range(self.nthreads):
            process = mp.Process(target=_play_n_games, args=(args[i],))
            processes.append(process)
            process.start()
            # if i % 8 == 7:
            #     sleep(1.5)
            # else:
            #     sleep(.1)
        threads_finished = 0
        ngames_finished = 0
        thread_pb = tqdm(total=self.nthreads, desc="Threads")
        ngames_pb = tqdm(total=ngames, desc="Games")
        while threads_finished < self.nthreads:
            partial_wins, partial_timers, done = self.results_queue.get()
            if done:
                threads_finished += 1
                thread_pb.update(1)
            wins += partial_wins
            timers += partial_timers
            ngames_pb.update(sum(partial_wins))
        for p in processes:
            p.join()
        for gpu_mgr in self.gpu_managers:
            gpu_mgr.stop()
        wins_ratio = wins / ngames
        time_per_move = timers / self.nthreads
        wins_pct = wins_ratio * 100
        # thread_pb.refresh()
        thread_pb.close()
        # ngames_pb.refresh()
        ngames_pb.close()
        if show_info:
            for i, player_name in enumerate(player_names):
                print(f"Player {player_name}: won {wins_pct[i]:.1f}% of games, spent {time_per_move[i]*1000:.0f}ms per move")
        return wins_pct, time_per_move

    def run(self, ngames, return_results_and_fig=False):
        start = time()
        print("-------------------")
        print("Running competition")
        print(f"Will be playing {ngames} games")
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            print(f"Player {i+1}: {name}")
            print(f"    has config {clip_cfg(cfg)}")
        combs = list(combinations(range(self.nplayers), self.gamemode))
        all_results = np.zeros((self.nplayers, self.nplayers))
        times_per_move = np.zeros(self.nplayers)
        for rnd, comb in enumerate(combs):
            print("\n===================")
            print(f"Starting round {rnd+1}")
            player_names = [self.player_names[_] for _ in comb]
            player_configs = [self.player_configs[_] for _ in comb]
            wins_pct, partial_times_per_move = self.play_round(player_names, player_configs, ngames, show_info = len(combs) != 1)
            if self.gamemode == 2:
                all_results[comb], all_results[comb[::-1]] = wins_pct
            for i in range(self.gamemode):
                times_per_move[comb[i]] += partial_times_per_move[i]
            print(f"Round {rnd+1} ended")
            print("==================\n")
        print("-------------------")
        print("Competition ended")
        print("Results summary:")
        if len(combs) > 1:
            print(f"  Games per round: {ngames}")
            print(f"  Number of  rounds: {len(combs)}")
        times_per_move /= (self.nplayers - 1)
        index_names = []
        for i, name in enumerate(self.player_names):
            index_names.append(f"{i+1}: {name}")
        if self.gamemode == 2:
            cpt_data = pd.DataFrame(all_results, columns=index_names, index=index_names)
        else:
            cpt_data = pd.DataFrame(wins_pct, index=index_names, columns=["Win Ratio [%]"]).T

        print("    Average time spent per move:")
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            print(f"\tPlayer {name}: {times_per_move[i]*1000:.0f}ms")
            if len(combs)>1:
                print(f"\t    has config {clip_cfg(cfg)}")
        print("Win Ratios [%]:")
        print(cpt_data)
        if self.gamemode == 2:
            for player in range(self.nplayers):
                dominant = True
                for opp in range(self.nplayers):
                    if opp == player:
                        continue
                    if all_results[player, opp] <= 50:
                        dominant = False
                        break
                if dominant:
                    dominant_player = player
                    break
        else:
            dominant = True
            dominant_player = np.argmax(wins_pct)
        if dominant:
                print(f"Player {self.player_names[dominant_player]} won the tournament")
                print(f"    has config {clip_cfg(self.player_configs[dominant_player])}")
        else:
            print(f"There is no clear tournament winner")

        if self.gamemode == 2:
            plt.figure(figsize=(2*self.nplayers,2*self.nplayers + 1.2))
        else:
            plt.figure(figsize=(2*self.nplayers+.3, self.nplayers+1.2))
        plt.title("Competition Results", fontsize=16)
        sns.heatmap(cpt_data, annot=True, cmap = 'YlGnBu', cbar = self.gamemode == 2,cbar_kws = {'label': 'Win Rate (%)'}, square=True)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.yticks(rotation=0)
        if self.gamemode == 2:
            plt.xlabel("Opponent", fontsize=13)
            plt.ylabel("Player", fontsize=13)
        else:
            plt.xlabel("Player", fontsize=13)
        s = f"Number of rounds: {len(combs)}\n"
        s += f"Number of games per round: {ngames}\n"
        s += "Contestants:\n"
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            s += f"    Player {i+1}: {name}; has config {clip_cfg(cfg)}; time/move: {times_per_move[i]*1000:.0f}ms\n"

# 2. Adjust the subplot to leave room at the bottom
        plt.subplots_adjust(bottom=0.25) # Increase this if the text is long

# 3. Add the text box
# x=0.5 (center), y=0.05 (near bottom edge of figure)
        plt.text(0.02, 0.02, s, 
                 fontsize=10,
                 family="monospace",
                 ha='left',      # Horizontal alignment
                 va='bottom',      # Vertical alignment
                 linespacing=1.5,
                 wrap=True,        # Helps if the string is incredibly long
                 transform = plt.gcf().transFigure,
                 bbox=dict(        # The "Auto-fit" box
                     boxstyle='round,pad=0.8', 
                     facecolor='white', 
                     edgecolor='none', 
                     alpha=0.9
                 ))
        
        duration = time() - start
        print(f"Competition lasted for {duration:.1f} seconds")
        print("-------------------")
        if return_results_and_fig:
            fig = plt.gcf()
            return all_results, fig
        fname = get_fname(self.player_names, self.player_configs, COMPETITION_FIG_DIR, "png")
        checkfn(fname)
        plt.savefig(fname)
        print(f"Saved competition results as '{fname}'")
        plt.close('all')

def main():
    nworkers = 16
    games_per_worker = 300
    worker_batch_size = 24
    ngames = nworkers * games_per_worker
    # model_path = "weights/evo-0_cycle-0.pth"
    # model_path = "weights/no_scaling.pth"
    # model_path = "weights/two_player.pth"
    model_path = "weights/four_player_updated.pth"
    tf_model_path = "weights/two_player_tf.weights.h5"
    # gpu_manager, gpu_manager_info = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=model_path)
    # gpu_manager_2, gpu_manager_info_2 = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=model_path)
    # player_names = ["TSPEnv", "NN", "Smart", "Eager", "Random"]
    player_names = [
            # "TSPEnv", 
            # "VNNMCTS",
            # "VNNMCTS",
            # "NNMCTS",
            # "NN",
            # "TSPEnv", 
            "Eager",
            # "TSPDet",
            # "TSPDet",
            # "TSPDet",
            # "TSPDet",
            # "Smart", 
            # "Smart"
            "Smart",
            "Random",
            "Random",
    ]
    player_configs = [
            # {},
            # {"maxdepth": 3},
            # {"fname": tf_model_path, "hyperparams": {"maxit":50}, "nplayers": 2, "use_tensorflow":True, "test_tf_loader":True},
            # {"hyperparams": {"maxit": 500, "num_sims": worker_batch_size, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info},
            # {"hyperparams": {"maxit": 500, "num_sims": worker_batch_size, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info_2},
            # {"hyperparams": {"maxit": 50, "num_sims": worker_batch_size, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info},
            # {"fname": model_path, "nplayers": 4, "use_tensorflow": False, "test_tf_loader": False, "hyperparams": {"maxit": 100}},
            # {"maxdepth": 8},
            {"maxdepth": 3, "maxit": 10},
            # {"maxdepth": 3, "maxit": 30},
            # {"maxdepth": 3, "maxit": 50},
            # {"maxdepth": 3, "maxit": 100},
            # {"maxdepth": 4},
            # {"maxdepth": 2, "maxit": 20},
            # {"fname": "weights/tw_ts_md_4_det.pth"},
            {},
            {},
            {},
    ]

    cpt = Competition(player_names, player_configs, gamemode=4, nthreads=nworkers, gpu_managers=[])
    cpt.run(ngames)


if __name__ == "__main__":
    main()

