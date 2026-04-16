import numpy as np
import multiprocessing
import time
import torch
import torch.multiprocessing as mp
import traceback
import os
from tqdm import tqdm

from tock import make
from players import get_players, get_fname, clip_cfg
from network import PolicyNN
from utils import getstate, revert_action_prob, checkfn
from data import GameData
from gpu_manager import get_gpu_manager

def choose_action_with_temperature(actions, action_prob, tau):
    # print(f"Tau is: {tau}")
    prob = action_prob ** (1/tau)
    prob = prob / np.sum(prob)
    action_idx = np.random.choice(np.arange(len(actions)), p=prob)
    best_action = actions[action_idx]
    return best_action


def _play_and_gen_data(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    player_names, player_configs, ngames, playparam, pid, results_queue = args
    e_threads = list(range(4, 32))
    os.sched_setaffinity(0, set(e_threads))
    for cfg in player_configs:
        if 'gpu_manager_info' in cfg:
            cfg['worker_id'] = pid

    tau, best_play_move = playparam["tau"], playparam["best_play_move"]
    nplayers = len(player_names)
    players, types = get_players(player_names, player_configs)
    env = make(nplayers)
    prob_list_total = []
    pos_list_total = []
    value_targets_total = []
    checkpoint = max(1, ngames // 10)
    start_interval = time.time()
    interval_ngames = 0
    for game in range(ngames):
        prob_list = []
        pos_list = []
        player_list = []
        obs, rew, done, info = env.reset()
        move = 0

        while not done:
            player_idx = info['player']
            player = players[info['player']]
            if types[player_idx] == 'environment':
                action_prob = player(env)
            elif types[player_idx] == 'obsinfo':
                action_prob = player(obs, info)
            if move >= best_play_move:
                action = info['space'][np.argmax(action_prob)]
            else:
                action = choose_action_with_temperature(info['space'], action_prob, tau)
            reverted_prob = revert_action_prob(action_prob, info)

            prob_list.append(reverted_prob)
            pos_list.append((getstate(obs, info)))
            player_list.append(info['player'])
            
            obs, rew, done, info = env.step(action=action)
            move += 1
        winner = info['winner']
        if nplayers == 2:
            value_targets = [1 if winner == player_list[i] else -1 for i in range(move)]
        else:
            value_targets = [np.array([1 if winner == (i + player_list[j]) % nplayers else 0 for i in range(nplayers)]) for j in range(move)]
        interval_ngames += 1
        prob_list_total += prob_list
        pos_list_total += pos_list
        value_targets_total += value_targets
        if time.time() - start_interval > .1:
            results_queue.put((interval_ngames, pos_list_total, prob_list_total, value_targets_total,False))
            prob_list_total = []
            pos_list_total = []
            value_targets_total = []
            start_interval = time.time()
            interval_ngames = 0


    results_queue.put((interval_ngames, pos_list_total, prob_list_total, value_targets_total,True))


class DataGenerator:
    def __init__(self, player_names, player_configs, playparam, *, nthreads, gpu_manager=None):
        self.playparam = playparam
        self.data = GameData()
        self.player_names = player_names
        self.player_configs = player_configs
        self.nplayers = len(self.player_names)
        self.nthreads = nthreads
        self.gpu_manager = gpu_manager

    def generate(self, ngames, return_data=False):
        self.results_queue = multiprocessing.Queue()
        start = time.time()
        print(f"====================================")
        print(f"Generating training data ({ngames} games with {self.nthreads} threads)")
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            print(f"Player {i+1}: {name}")
            print(f"    has config {clip_cfg(cfg)}")

        games_per_core = max(1, ngames // self.nthreads)
        leftovers = ngames % self.nthreads
        if games_per_core == 1:
            leftovers = 0
            ngames = self.nthreads
        args = [(self.player_names, self.player_configs, games_per_core if (_!=0) else games_per_core + leftovers, self.playparam, _, self.results_queue) for _ in range(self.nthreads)]
        threads_finished = 0
        need_gpu_manager = False
        for cfg in self.player_configs:
            if 'gpu_manager_info' in cfg:
                need_gpu_manager = True
        if need_gpu_manager:
            self.gpu_manager.start()
        processes = []
        print(f"Launching {self.nthreads} workers")
        for i in range(self.nthreads):
            process = mp.Process(target=_play_and_gen_data, args=(args[i],))
            processes.append(process)
            process.start()
        threads_finished = 0
        thread_pb = tqdm(total=self.nthreads, desc="Threads")
        games_pb = tqdm(total=ngames, desc="Games")
        samples_pb = tqdm(total=float("inf"), desc="Samples")
        while threads_finished < self.nthreads:
            interval_ngames, pos_list, prob_list, value_list, done = self.results_queue.get()
            samples_pb.update(len(pos_list))
            games_pb.update(interval_ngames)

            if done:
                threads_finished += 1
                thread_pb.update(1)
            for pos, prob, value in zip(pos_list, prob_list, value_list):
                self.data.append(*pos, prob, value)

        for p in processes:
            p.join()
        if need_gpu_manager:
            self.gpu_manager.stop()
        # thread_pb.refresh()
        # games_pb.refresh()
        # samples_pb.refresh()
        thread_pb.close()
        games_pb.close()
        samples_pb.close()
        print(f"Finished generating training data")
        print(f"Now {len(self.data)} samples in database.")
        duration = time.time() - start
        print(f"Spent {duration:.1f} seconds generating data")
        print(f"====================================")

        if return_data:
            return self.data
        fname = get_fname(self.player_names[:1], self.player_configs[:1], "data", extra_info=f"ngames_{ngames}_nplayers_{self.nplayers}")
        try: 
            self.data.save(fname)
        except:
            traceback.print_exc()
            breakpoint()

def main():
    # player_names = ["TSPEnv", "NN", "Smart", "Eager", "Random"]
    nworkers = 16
    games_per_worker = 50
    worker_batch_size = 32
    ngames = nworkers * games_per_worker
    model_path = "weights/tw_ts_md_4_det.pth"
    gpu_manager, gpu_manager_info = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=model_path)
    player_names = [
            # "TSPEnv", 
            # "VNNMCTS",
            # "NNMCTS",
            # "NN",
            "TSPEnv", 
            "TSPEnv", 
            "TSPEnv",
            "TSPEnv",
            # "TSPDet",
            # "TSPDet",
    ]
    player_configs = [
            # {"maxdepth": 5},
            # {"fname": "weights/tw_ts_md_4_det.pth", "hyperparams": {"maxit":1}, "return_type": "probabilities"},
            # {"hyperparams": {"maxit": 1600, "num_sims": WORKER_BATCH_SIZE, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info},
            # {"maxdepth": 5, "return_type": "probabilities", "maxit": 50},
            # {"maxdepth": 5, "return_type": "probabilities", "maxit": 50},
            # {"fname": "weights/tw_ts_md_4.pth"},
            {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
            {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
            {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
            {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
    ]
    playparam = {"tau": 1, "best_play_move": 10}
    generator = DataGenerator(player_names, player_configs, playparam, nthreads=nworkers, gpu_manager=gpu_manager)
    generator.generate(ngames)
    # generator.data.show_random(100)


if __name__ == "__main__":
    main()
