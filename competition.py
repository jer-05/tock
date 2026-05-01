import torch
import threading
import numpy as np
import os
import multiprocessing
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib
import traceback
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time, sleep
import inspect
from tqdm import tqdm
import torch.multiprocessing as mp

from elo_manager import EloManager
from tock import make, AVERAGE_GAME_LENGTH
from fasttock import FastTockGame
from gpu_manager import get_gpu_manager
from players import get_players, get_fname, get_fullnames, clip_cfg
from network import PolicyNN
from utils import getstate, checkfn, remove_dirs, clear_lines, get_ort_inference_session, check_abortion
import cProfile
import queue

COMPETITION_FIG_DIR = "figures/competition"

def get_time_str(time):
    if time * 1000 < 3:
        timestr = f"{time*1000000:.0f}ns"
    elif time < 3:
        timestr = f"{time*1000:.0f}ms"
    else:
        timestr = f"{time:.0f}s"
    return timestr

def _play_n_games(args):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        player_names, player_configs, ngames, pid, results_queue, stopevent, ort_info = args
        e_threads = list(range(4, 32))
        os.sched_setaffinity(0, set(e_threads))
        for cfg in player_configs:
            if 'gpu_manager_info' in cfg or 'fname' in cfg:
                cfg['worker_id'] = pid
            if 'ort_info' in cfg:
                cfg['ort_info'] = ort_info
        def _offs_arr(arr, offs):
            return [arr[(i + offs) % len(arr)] for i in range(len(arr))]
        players, types = get_players(_offs_arr(player_names, pid), _offs_arr(player_configs, pid))
        players = _offs_arr(players, -pid)
        types = _offs_arr(types, -pid)
        nplayers = len(player_names)
        env = make(nplayers)
        wins = np.zeros(len(players))
        timers = np.zeros(len(players))
        checkpoint = max(ngames // 10, 1)
        moves = np.zeros(len(players))
        start_wait = time()
        for game in range(ngames):
            # print(f"[{pid}]: starting game {game}", flush=True)
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
                moves[player_idx] += 1
            wins[(info['winner'] + game) % nplayers] += 1
            if (time() - start_wait > .1) or game == ngames -1:
                # print(f"[{pid}]: putting {int(sum(wins))} results")
                if stopevent.is_set() or game == ngames - 1:
                    results_queue.put((wins, timers, moves, True))
                    # print(f"[{pid}]: stopped")
                    return
                else:
                    results_queue.put((wins, timers, moves, False))
                    sleep(.001)
                    wins = np.zeros(len(players))
                    timers = np.zeros(len(players))
                    start_wait = time()
                    moves = np.zeros(len(players))
    except Exception:
        print(f"[workerid {pid}]: _play_n_games: An error occured! Stopping now.")
        traceback.print_exc()


class Competition:
    def __init__(self, player_names, player_configs, gamemode = 2, nthreads=os.cpu_count() - 1, gpu_managers=[], ort_info = [], filemanager=None, elo_manager=None, verbose=False, idd=None, ladder_cpt = False, stopevent=None):
        self.gamemode = gamemode
        self.nthreads = nthreads
        self.ladder_cpt = ladder_cpt
        self.idd=idd
        self.filemanager=filemanager
        self.stopevent = stopevent
        self.elo_manager = elo_manager
        if self.elo_manager is not None:
            assert self.gamemode == 2, f"Competition: Cannot track Elo for 4/6 player"
        self.gpu_managers = gpu_managers
        self.ort_info = ort_info
        assert (self.gamemode in [2, 4, 6]), f"Invalid game mode '{self.gamemode}'"
        assert (len(player_names) == len(player_configs)), f"Missing or redundant player configurations"
        assert (len(player_names) != 1), f"Only one contestant"
        self.player_names = player_names
        self.nplayers = len(self.player_names)
        if self.gamemode in [4, 6] and ladder_cpt == False:
            raise NotImplementedError
        self.player_configs = player_configs
        self.clipped_configs = [clip_cfg(cfg) for cfg in player_configs]
        self.verbose = verbose
   
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
        stopevent = mp.Event()
        results_queue = mp.Queue()
        args = [(player_names, player_configs, games_per_core if (_!=0) else games_per_core + leftovers, _, results_queue, stopevent, self.ort_info) for _ in range(self.nthreads)]
        threads_finished = 0
        for gpu_mgr in self.gpu_managers:
            print(f"Starting GPU manager")
            gpu_mgr.start()
        processes = []
        print(f"Launching {self.nthreads} workers")
        for i in range(self.nthreads):
            process = mp.Process(target=_play_n_games, args=(args[i],))
            processes.append(process)
        abortion_monitor = threading.Thread(target=check_abortion, args=[stopevent], kwargs={"extra_stop_event": self.stopevent}, daemon=True)
        abortion_monitor.start()
        for p in processes:
            p.start()
        threads_finished = 0
        ngames_finished = 0
        thread_pb = tqdm(total=self.nthreads, desc="Threads")
        ngames_pb = tqdm(total=ngames, desc="Games")
        status = tqdm(total=1, bar_format="{desc}")
        time_status = tqdm(total=1, bar_format="{desc}")
        CIs = np.array([[0, 1] for i in range(self.gamemode)], dtype=float)
        moves = np.zeros(self.gamemode)
        while threads_finished < self.nthreads:
            try:
                # print(f"Getting results")
                partial_wins, partial_timers, partial_moves, done = results_queue.get(timeout=60)
            except queue.Empty:
                print(f"Competition: results queue timeout triggered!!!")
                break
            if done:
                # print(f"Done!")
                threads_finished += 1
                thread_pb.update(1)
            wins += partial_wins
            timers += partial_timers
            moves += partial_moves
            ngames_pb.update(int(sum(partial_wins)))
            s = ""
            time_s = ""
            n = sum(wins)
            for i, (wini, timer, _moves) in enumerate(zip(wins, timers, moves)):
                p = wini / n
                CI = self.get_95CI(n, p)
                CIs[i] = CI
                s += f"P{i+1}: {p*100:.1f}% [{CI[0]*100:.1f}, {CI[1]*100:.1f}] | "
                time_s += f"P{i+1}: {get_time_str(timer/_moves)} | "
            s = s[:-3]; time_s = time_s[:-3]
            status.set_description_str(s)
            time_status.set_description_str(time_s)
        stopevent.set()
        while not results_queue.empty():
            try:
                results_queue.get_nowait()
            except:
                break
        if sum(wins) < ngames:
            ngames = sum(wins)
            tqdm.write(f"Round was preemptively ended at {int(ngames)} games")
        for i, p in enumerate(processes):
            p.join(timeout = 10)
            if p.is_alive():
                print(f"Process {p.pid} timed out! Terminating...")
                p.terminate() 
                p.join()
        for gpu_mgr in self.gpu_managers:
            gpu_mgr.stop()
        wins_ratio = wins / ngames
        wins_pct = wins_ratio * 100
        time_per_move = timers / moves
        thread_pb.close()
        ngames_pb.close()
        status.close()
        time_status.close()
        if show_info:
            for i, (player_name, time, win_pct) in enumerate(zip(player_names, timers, wins_pct)):
                timestr = get_time_str(time)
                print(f"Player {player_name}: won {win_pct:.1f}% of games, spent {timestr} per move")
        if self.elo_manager is not None:
            self.elo_manager.add_match_result(player_names, player_configs, wins)
        return wins_pct, CIs * 100, time_per_move, ngames

    def get_95CI(self, n, p):
        inv_normal95 = 1.96
        var = n*p*(1-p)
        std = np.sqrt(var)
        std_p = std / n
        margin = inv_normal95 * std_p
        CI = np.array([max(0, p - margin), min(1, p + margin)])
        return CI

    def run(self, ngames, return_results_and_fig=False):
        start = time()
        print("-------------------")
        print("Running competition")
        if not self.ladder_cpt:
            combs = list(combinations(range(self.nplayers), self.gamemode))
        else:
            other_player_combs = list(combinations(range(1, self.nplayers), self.gamemode-1))
            combs = [[0, *comb] for comb in other_player_combs]
        print(f"Will be playing {len(combs) * ngames} games ({ngames} per round for {len(combs)} rounds)")
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            print(f"Player {i+1}: {name}")
            print(f"    has config {clip_cfg(cfg)}")
        if not self.ladder_cpt:
            all_results = np.zeros((self.nplayers, self.nplayers))
        else:
            all_results = np.zeros((len(combs), self.gamemode))
        times_per_move = np.zeros(self.nplayers)
        rds_ngames = []
        all_CIs = np.zeros((*all_results.shape, 2))
        for rnd, comb in enumerate(combs):
            print("\n===================")
            print(f"Starting round {rnd+1}")
            player_names = [self.player_names[_] for _ in comb]
            player_configs = [self.player_configs[_] for _ in comb]
            wins_pct, CIs, partial_times_per_move, rd_ngames = self.play_round(player_names, player_configs, ngames, show_info = len(combs) != 1)
            rds_ngames.append(int(rd_ngames))
            if not self.ladder_cpt:
                all_results[comb], all_results[comb[::-1]] = wins_pct
                all_CIs[comb], all_CIs[comb[::-1]] = CIs
            else:
                all_results[rnd] = wins_pct
                all_CIs[rnd] = CIs
            for i in range(self.gamemode):
                times_per_move[comb[i]] += partial_times_per_move[i]
            print(f"Round {rnd+1} ended")
            print("==================\n")
        print("-------------------")
        print("Competition ended")
        print("Results summary:")
        if len(combs) > 1:
            print(f"  Games per round: {rds_ngames}")
            print(f"  Number of  rounds: {len(combs)}")
        times_per_move /= (self.nplayers - 1)
        index_names = []
        column_names = []
        if self.ladder_cpt:
            for i in range(all_results.shape[0]):
                s = "0"
                for player_idx in combs[i][1:]:
                    s += f"v{player_idx}"
                index_names.append(s)
        else:
            for i, name in enumerate(self.player_names):
                column_names.append(f"{i+1}: {name}")
            index_names = column_names
        if column_names:
            cpt_data = pd.DataFrame(all_results, columns=column_names, index=index_names)
        else:
            cpt_data = pd.DataFrame(all_results, index=index_names)
        print("    Average time spent per move:")
        for i, (name, cfg, timer) in enumerate(zip(self.player_names, self.player_configs, times_per_move)):
            timestr = get_time_str(timer)
            print(f"Player {name}: {timestr} per move")
            if len(combs)>1:
                print(f"\t    has config {clip_cfg(cfg)}")
        print("Win Ratios [%]:")
        print(cpt_data)
        plt.figure(figsize=(cpt_data.shape[1] + 3, cpt_data.shape[0] + self.nplayers + 4))
        plt.title("Competition Results", fontsize=16)
        annot = [[""]*cpt_data.shape[1] for i in range(cpt_data.shape[0])]
        for i in range(cpt_data.shape[0]):
            for j in range(cpt_data.shape[1]):
                if not self.ladder_cpt:
                    if i == j:
                        annot[i][j] = "—"
                        continue
                s = f"{all_results[i][j]:.1f}\n"
                s += f"[{all_CIs[i][j][0]:.1f}, {all_CIs[i][j][1]:.1f}]"
                annot[i][j] = s

        sns.heatmap(cpt_data, annot=annot, cmap = 'YlGnBu', cbar = self.gamemode == 2,cbar_kws = {'label': 'Win Rate (%)'}, square=True, fmt="")
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.yticks(rotation=0)
        if not self.ladder_cpt:
            plt.xlabel("Opponent", fontsize=13)
            plt.ylabel("Player", fontsize=13)
        else:
            plt.xlabel("Player", fontsize=13)
        s = f"Number of rounds: {len(combs)}\n"
        s += f"Number of games for each round: {rds_ngames}\n"
        s += "Contestants:\n"
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            if times_per_move[i] * 1000 < 1:
                timestr = f"{times_per_move[i]*1000000:.0f}ns"
            else:
                timestr = f"{times_per_move[i]*1000:.0f}ms"
            s += f"    Player {i+1}: {name}; has config {clip_cfg(cfg)}; time/move: {timestr}\n"

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
        if self.filemanager is not None:
            fig = plt.gcf()
            if self.idd is not None:
                self.filemanager.save(fig, "cpt", idd=self.idd)
            else:
                self.filemanager.save(fig, "cpt")
            fig.clf()
            return
        fname = get_fname(self.player_names, self.player_configs, COMPETITION_FIG_DIR, ".png", extra_info=f"ngames_{round(sum(rds_ngames)/len(rds_ngames))}")
        checkfn(fname)
        plt.savefig(fname)
        print(f"Saved competition results as '{fname}'")
        plt.close('all')

def main(ort_info):
    nworkers=4 if os.cpu_count() == 8 else 16
    games_per_worker =300
    worker_batch_size = 32
    ngames = nworkers * games_per_worker
    # fname = "weights/2_players/datasize_59981_epochs_10_batch_128_lr_0.001_iteration_1.pth"
    fname = "weights/2_players/session39best.pth"
    fname = "weights/2_players/datasize_300005_epochs_10_batch_256_lr_0.001_iteration_1.pth"
    # gpu_manager, gpu_manager_info = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=model_path)
    player_names = [
            # "TSPEnv", 
            "Smart", 
            # "Random",
            "NN",
            # "NNMCTS",
            # "TSPDet",
    ]
    player_configs = [
            # {"maxdepth": 5},
            {},
            {"fname": fname, "ort_info": ""},
            # {"fname": fname, "hyperparams": {"maxit": 200, "cpuct": 2, "epsilon": 25, "alpha":.2, "noise": False}, "ort_info":''},
            # {"maxdepth": 5, "maxit": 40},

    ]
    cpt = Competition(player_names, player_configs, gamemode=2, nthreads=nworkers, ort_info=ort_info, elo_manager=None, verbose=False, ladder_cpt=False)
    cpt.run(ngames)


if __name__ == "__main__":
    from mp_ort_import import exec_main
    exec_main(main)

