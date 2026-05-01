import numpy as np
import multiprocessing
import time
import torch
import traceback
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import queue

from tock import make
from fasttock import action_tuple_to_action_number, action_number_to_action_tuples, tupled_action_space_to_action_mask
from players import get_players, get_fname, clip_cfg
from network import PolicyNN
from utils import getstate, checkfn, check_abortion
from data import GameData
from gpu_manager import get_gpu_manager
import threading
import signal
import random

def choose_action_with_temperature(actions, action_prob, tau):
    prob = action_prob ** (1/tau)
    prob = prob / np.sum(prob)
    action_idx = np.random.choice(np.arange(len(actions)), p=prob)
    best_action = actions[action_idx]
    return best_action


def _play_and_gen_data(args):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        player_names, player_configs, ngames, playparam, pid, results_queue, stopevent, ort_info = args
        e_threads = list(range(4, 32))
        os.sched_setaffinity(0, set(e_threads))
        return_types = []
        for cfg in player_configs:
            if 'gpu_manager_info' in cfg or 'fname' in cfg:
                cfg['worker_id'] = pid
            if 'ort_info' in cfg:
                cfg['ort_info'] = ort_info
            if 'return_type' not in cfg or cfg['return_type'] == 'best_action':
                return_types.append('best_action')
            elif cfg['return_type'] == 'unique_probabilities':
                return_types.append('unique_probabilities')
            else:
                assert False, f"Invalid return type {cfg['return_type']}"
        # tqdm.write(f"Using player return types: {return_types}")

        tau, best_play_move = playparam["tau"], playparam["best_play_move"]
        nplayers = len(player_names)
        def _offs_arr(arr, offs):
            return [arr[(i + offs) % len(arr)] for i in range(len(arr))]
        players, types = get_players(_offs_arr(player_names, pid), _offs_arr(player_configs, pid))
        players = _offs_arr(players, -pid)
        types = _offs_arr(types, -pid)
        _player_names = []
        for name, config, return_type in zip(player_names, player_configs, return_types):
            idstr = f""
            if "fname" in config:
                idstr += f"_fname_{config['fname'][-10:]}"
            if "maxdepth" in config:
                idstr += f"_maxdepth_{config['maxdepth']}"
            if "maxit" in config:
                idstr += f"_maxit_{config['maxit']}_"
            if "hyperparameters" in config:
                if "maxit" in config["hyperparameters"]:
                    idstr += f"_maxit_{config['hyperparameters']['maxit']}"
            if return_type == "best_action":
                move_str = "best_action"
            elif return_type == "unique_probabilities":
                move_str = f"prob_bpmove_{best_play_move:.0f}_tau_{tau:.1f}"
            player_name = f"{name}_{move_str}{idstr}"
            _player_names.append(player_name)
        player_names = _player_names
        env = make(nplayers)
        prob_list_total = []
        pos_list_total = []
        value_targets_total = []
        moves_total = []
        names_total = []
        game_lengths_total = []
        checkpoint = max(1, ngames // 10)
        start_interval = time.time()
        interval_ngames = 0
        for game in range(ngames):
            prob_list = []
            pos_list = []
            player_list = []
            names_list=[]
            obs, rew, done, info = env.reset()
            move = 0

            while not done:
                player_idx = (info['player'] + game) % nplayers
                player = players[player_idx]
                names = [player_names[(player_idx + i) % nplayers] for i in range(nplayers)]
                # tqdm.write(f"Now player {player_names[player_idx]}'s move")
                if types[player_idx] == 'environment':
                    res = player(env)
                elif types[player_idx] == 'obsinfo':
                    res = player(obs, info)
                else:
                    # tqdm.write(f"_play_and_gen_data: Invalid player type '{types[player_idx]}'")
                    breakpoint()

                action_prob = np.zeros(50)
                if return_types[player_idx] == 'unique_probabilities':
                    masked_action_prob, action_numbers = res
                    action_prob[action_numbers] = masked_action_prob
                    if move >= best_play_move:
                        action_number = action_numbers[np.argmax(masked_action_prob)]
                    else:
                        action_number = choose_action_with_temperature(action_numbers, masked_action_prob, tau)
                    actions = action_number_to_action_tuples(action_number, sorted(obs[info['player']]), info['cards'])
                    fastgame_action = random.choice(actions)
                    argorder = np.argsort(np.array(obs[info['player']]))
                    action = (fastgame_action[0], argorder[fastgame_action[1]])
                    # tqdm.write(f"Got action_prob {action_prob}")
                    # tqdm.write(f"Choosing action {action}")
                elif return_types[player_idx] == 'best_action':
                    action = res
                    action_prob = tupled_action_space_to_action_mask([action], obs[info['player']], info['cards']).astype(float)
                else:
                    tqdm.write(f"_play_and_gen_data: Invalid return type '{return_types[player_idx]}'")
                    breakpoint()
                if np.equal(action_prob, np.zeros(50)).all():
                    print(f"_play_and_gen_data: Invalid action prob: actions are all zero!")
                    print(f"names: {names}")
                    print(f"return type: {return_types[player_idx]}")
                    breakpoint()
                prob_list.append(action_prob)
                state = getstate(obs, info, fastgame_type=False)
                pos_list.append(state)
                player_list.append(info['player'])
                names_list.append(names)
                # print(f"Appended data to queue:")
                # print(f"  action_prob: {action_prob}")
                # print(f"  names      : {names}", flush=True)
                
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
            moves_arr = list(range(move))
            game_lengths_arr = [move-1] * move
            moves_total += moves_arr
            names_total += names_list
            game_lengths_total += game_lengths_arr
            if time.time() - start_interval > .1:
                if stopevent.is_set():
                    results_queue.put((interval_ngames, pos_list_total, prob_list_total, value_targets_total, moves_total, game_lengths_total, names_total, True))
                    return
                results_queue.put((interval_ngames, pos_list_total, prob_list_total, value_targets_total, moves_total, game_lengths_total, names_total, False))
                time.sleep(.001)
                prob_list_total = []
                pos_list_total = []
                value_targets_total = []
                moves_total = []
                game_lengths_total = []
                names_total = []
                start_interval = time.time()
                interval_ngames = 0


        results_queue.put((interval_ngames, pos_list_total, prob_list_total, value_targets_total,moves_total, game_lengths_total, names_total, True))
    except Exception:
        traceback.print_exc()
        return


class DataGenerator:
    def __init__(self, player_names, player_configs, playparam, ort_info, *, nthreads, gpu_manager=None, stopevent=None):
        self.playparam = playparam
        self.data = GameData(nplayers=len(player_names))
        self.player_names = player_names
        self.player_configs = player_configs
        self.nplayers = len(self.player_names)
        self.nthreads = nthreads
        self.gpu_manager = gpu_manager
        self.ort_info = ort_info
        self.stopevent=stopevent

    def generate(self, ngames, *, nsamples=np.inf, return_data=False):
        results_queue = multiprocessing.Queue()
        start = time.time()
        print(f"====================================")
        print(f"Generating training data ({ngames} games with {self.nthreads} threads)")
        if nsamples is not None:
            print(f"Sample cap is {nsamples} (target)")
        for i, (name, cfg) in enumerate(zip(self.player_names, self.player_configs)):
            print(f"Player {i+1}: {name}")
            print(f"    has config {clip_cfg(cfg)}")

        games_per_core = max(1, ngames // self.nthreads)
        leftovers = ngames % self.nthreads
        if games_per_core == 1:
            leftovers = 0
            ngames = self.nthreads
        stopevent = mp.Event()
        args = [(self.player_names, self.player_configs, games_per_core if (_!=0) else games_per_core + leftovers, self.playparam, _, results_queue, stopevent, self.ort_info) for _ in range(self.nthreads)]
        threads_finished = 0
        need_gpu_manager = False
        for cfg in self.player_configs:
            if 'gpu_manager_info' in cfg:
                need_gpu_manager = True
        if need_gpu_manager:
            self.gpu_manager.start()
        processes = []
        abortion_manager = threading.Thread(target=check_abortion, args=[stopevent],kwargs={"extra_stop_event": self.stopevent}, daemon=True)
        print(f"Launching {self.nthreads} workers")
        for i in range(self.nthreads):
            process = mp.Process(target=_play_and_gen_data, args=(args[i],))
            processes.append(process)
            process.start()
        abortion_manager.start()
        threads_finished = 0
        thread_pb = tqdm(total=self.nthreads, desc="Threads")
        games_pb = tqdm(total=ngames, desc="Games")
        samples_pb = tqdm(total=nsamples, desc="Samples")
        cap_reached = False
        while threads_finished < self.nthreads:
            try:
                # print(f"Getting results")
                interval_ngames, pos_list, prob_list, value_list, move_list, l_list, names_list, done = results_queue.get(timeout=20)
            except queue.Empty:
                print(f"DataGenerator: results queue timeout triggered!!!")
                break
            samples_pb.update(len(pos_list))
            games_pb.update(interval_ngames)
            if done:
                threads_finished += 1
                thread_pb.update(1)
            assert len(pos_list) == len(prob_list) == len(value_list) == len(move_list) == len(l_list)
            for pos, prob, value, move, l, names in zip(pos_list, prob_list, value_list, move_list, l_list, names_list):
                self.data.append(*pos, prob, value, move, l, names)
            if samples_pb.n > nsamples:
                cap_reached = True
                stopevent.set()
        if cap_reached:
            tqdm.write(f"Aborted generation of data samples at data sample cap of {nsamples}")
        elif stopevent.is_set():
            tqdm.write(f"Aborted data generation by user input")
        stopevent.set()

        while not results_queue.empty():
            try:
                results_queue.get_nowait()
            except:
                break
        for i, p in enumerate(processes):
            p.join(timeout = 10)
            if p.is_alive():
                print(f"Process {p.pid} timed out! Terminating...")
                p.terminate() 
                p.join()
        if need_gpu_manager:
            self.gpu_manager.stop()
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
        fname = get_fname(self.player_names, self.player_configs, f"data/{self.nplayers}_players", extension="", extra_info=f"size_{len(self.data)}")
        if len(self.data) > 0:
            self.data.save(fname)
            return fname
        else:
            print(f"Failed generating any data")
            return None

def main(ort_info):
    # player_names = ["TSPEnv", "NN", "Smart", "Eager", "Random"]
    nworkers=4 if os.cpu_count() == 8 else 16
    nworkers = 4
    games_per_worker = 150
    worker_batch_size = 32
    ngames = nworkers * games_per_worker
    model_path = "weights/two_player.pth"
    # gpu_manager, gpu_manager_info = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=model_path)
    player_names = [
            # "TSPEnv", 
            # "VNNMCTS",
            # "NNMCTS",
            # "NNMCTS",
            # "NN",
            # "TSPEnv", 
            "Random",
            "Capture",
            "Smart",
            "TSPEnv", 
            # "TSPEnv",
            # "TSPEnv",
            # "TSPDet",
            # "TSPDet",
    ]
    player_configs = [
            # {"maxdepth": 5},
            # {"fname": "weights/two_player_copy.pth", "hyperparams": {"maxit": 100, "cpuct": 2}, "ort_info":'', "return_type": "probabilities"},
            # {"fname": model_path, "hyperparams": {"maxit": 20, "cpuct": 2, "epsilon": 25, "alpha":.2, "noise": False}, "ort_info":'', "return_type": "probabilities", "verbose": False},
            # {"fname": model_path, "hyperparams": {"maxit": 20, "cpuct": 2, "epsilon": 25, "alpha":.2, "noise": False}, "ort_info":'', "return_type": "probabilities", "verbose": False},
            # {"fname": "weights/tw_ts_md_4_det.pth", "hyperparams": {"maxit":1}, "return_type": "probabilities"},
            # {"hyperparams": {"maxit": 1600, "num_sims": WORKER_BATCH_SIZE, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info},
            {},
            {},
            {},
            {},
            # {"maxdepth": 3, "return_type": "unique_probabilities"},
            # {"maxdepth": 5, "return_type": "unique_probabilities"},
            # {"maxdepth": 5, "return_type": "probabilities", "maxit": 50},
            # {"fname": "weights/tw_ts_md_4.pth"},
            # {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
            # {"maxdepth": 1, "return_type": "best_action", 'store_rng': False},
            # {"maxdepth": 1, "return_type": "best_action", 'store_rng': False},
            # {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
            # {"maxdepth": 1, "return_type": "probabilities", 'store_rng': False},
    ]
    playparam = {"tau": 1, "best_play_move": 10}
    generator = DataGenerator(player_names, player_configs, playparam, nthreads=nworkers, ort_info=ort_info)
    datafname = generator.generate(ngames)
    data = generator.data
    data.show_random()
    data.save_statistics()


if __name__ == "__main__":
    from mp_ort_import import exec_main
    exec_main(main)
