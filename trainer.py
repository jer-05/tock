import torch
import numpy as np
import os
import time
import traceback
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import multiprocessing
import torch.multiprocessing as mp
import random
import pickle
import copy
import threading

from tock import make, DEAL_ORDER, AVERAGE_GAME_LENGTH
from fasttock import AVERAGE_GAME_LENGTHS
from network import PolicyNN
from utils import remove_dirs, clear_lines, print_block, print_important, print_v, check_abortion
from data import GameData
from players import NNPlayer, clip_cfg
from mcts import MCTS, Node, totorch
from competition import Competition
from datagen import DataGenerator
from filemanager import TrainingFileManager
from hyperparameter_manager import HyperparameterManager
from elo_manager import EloManager
from nn_trainer import NNTrainer

device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

WEIGHTS_DIR = "weights"
TRAINING_FIG_DIR = "figures/training"

DEFAULT_PLAYER_NAMES = [
        "Random",
        "Eager",
        "Smart",
        "TSPDet",
        "TSPEnv",
]
DEFAULT_PLAYER_CONFIGS = [
        {},
        {},
        {},
        {"maxdepth": 4, "maxit": 30}, 
        {"maxdepth": 7}
]

class Trainer():
    def __init__(self, *, hyperparams, ort_info, old_session_idx=None, verbose=False):
        print_v("Initializing trainer")
        self.ort_info = ort_info
        self.stopevent = mp.Event()
        self.verbose = verbose
        self.nplayers = hyperparams["nplayers"]
        self.tfm = TrainingFileManager(old_session_idx, verbose=self.verbose)
        hpp_fname = self.tfm.getlatest("hppdump", soft_fail=True)
        if hpp_fname is not None and hyperparams is None:
            print("Using previously saved hyperparameters")
            with open(hpp_fname, "rb") as f:
                hyperparams = pickle.load(f)
        else:
            if hyperparams is None:
                print_important("Using dummy hyperparameters")
                with open("hyperparameters/dummy.pkl", "rb") as f:
                    hyperparams = pickle.load(f)
            else:
                print("Using hyperparameters passed to function call")
            self.tfm.save(hyperparams, "hpp")

        self.hpm = HyperparameterManager(hyperparams, ncycle_not_evolved=self.tfm.cycle_ctr)
        self.nw = PolicyNN(self.nplayers)
        em_dump_fname = self.tfm.getlatest("elodump", soft_fail=True)
        if em_dump_fname is None:
            self.em = EloManager(nplayers=self.nplayers, verbose=self.verbose, challenger_id={"fname": "current"})
        else:
            with open(em_dump_fname, "rb") as f:
                self.em = pickle.load(f)
        self.always_evolve = False
        if "always_evolve" in hyperparams and hyperparams["always_evolve"]==True:
            self.always_evolve = True
            print(f"Always evolving !!!")

        self.data = GameData(nplayers=self.nplayers)
        self.data.set_buffer_size(self.hpm["replay_buffer_size"])
        if (datafname:=self.tfm.getlatest("data", soft_fail=True)) is not None:
            print(f"Loading saved replay buffer")
            self.data.load_dir(datafname)
            agefname = self.tfm.getlatest("dataage")
            print(f"Loading saved data age file")
            self.data_ages = torch.load(agefname, weights_only=False)
            self.data_age = min(self.data_ages) - 1
        else:
            self.data_age = 0
            self.data_ages = []

        if self.tfm.evo_ctr == 0 and self.tfm.cycle_ctr == 0:
            self.tfm.save(self.nw.state_dict(), "wcurr", idd="start_weights")
        self.nn_trainer = NNTrainer(self.nw, self.data, data_ages=self.data_ages, filemanager=self.tfm)


    def gen_train_data(self):
        print(f"Starting self-play")
        former_len = len(self.data)
        print(f"Already {former_len} samples in database")
        if self.hpm.switch_datagen_model():
            print(f"Model did not improve for self.{self.tfm.cycle_ctr} self-play cycles, so using best model for data generation")
            model_path = self.tfm.getlatest("wbest")
        else:
            model_path = self.tfm.getlatest("wcurr")
        generator = DataGenerator(*self.get_players([model_path] * self.nplayers, False), self.hpm["play_params"], nthreads=self.hpm["nworkers"], ort_info=self.ort_info, stopevent=self.stopevent)
        gen_epochs = self.hpm["self_play_epochs_per_cycle"] / self.hpm["train_play_rounds_per_cycle"]
        gen_samples = round(gen_epochs * self.hpm["replay_buffer_size"])
        appr_gen_games = round(1.5 * gen_samples / AVERAGE_GAME_LENGTHS[str(self.nplayers)])
        data = generator.generate(appr_gen_games, nsamples = gen_samples, return_data=True)
        self.data_ages += [self.data_age] * len(data)
        self.data_age -= 1
        self.data.load_data(data, preserve_buffer_size=True)
        self.tfm.save(self.data, "data")
        self.tfm.save(self.data_ages, "dataage")
        assert self.data_age == min(self.data_ages) - 1

    def train(self):
        print_v("Training started")
        self.stopevent = mp.Event()
        stopper = threading.Thread(target=check_abortion, args=[self.stopevent], kwargs={"reset":True}, daemon=True)
        stopper.start()
        try:
            rounds_per_cycle = self.hpm["train_play_rounds_per_cycle"]
            while not self.hpm.stop_training() and not self.stopevent.is_set():
                print_block(f"Evolution {self.tfm.evo_ctr} | Cycle {self.tfm.cycle_ctr}")
                print(self.hpm)
                while self.tfm.round_ctr < rounds_per_cycle and not self.stopevent.is_set():
                    print_block(f"Round {self.tfm.round_ctr}")
                    self.gen_train_data()
                    self.nn_trainer.train_nn(
                        self.hpm["train_epochs"],
                        self.hpm["lr"], 
                        self.hpm["train_batch_size"], 
                        gamma = self.hpm["sampler_gamma_decay"],
                        min_value_loss_weight=self.hpm["min_value_loss_train_weight"]
                    )
                    self.tfm.plusround()
                if self.stopevent.is_set():
                    break
                self.nn_trainer.plot_losses_accs(plotall=True)
                self.evolve()
        except (KeyboardInterrupt, EOFError, ConnectionResetError, BrokenPipeError):
            pass
        if not self.hpm.stop_training(verbose=False):
            print_important("Interrupted training")
        self.em.update_elo()
        self.tfm.save(self.em, "elo")
        self.model_evolution_benchmark()
        print_block("Training ended")

    def evolve(self):
        if self.tfm.getlatest("wbest", soft_fail=True) is None:
            print("No best model yet, setting current to best by default")
            self.tfm.save(self.nw.state_dict(), "wbest", idd="best")
            evolved = True
        else:
            print("Comparing current model to latest best model")
            best_model_fname = self.tfm.getlatest("wbest")
            current_model_fname = self.tfm.getlatest("wcurr")
            [player_name_best, player_name_curr], [player_config_best, player_config_curr] = self.get_players([best_model_fname, current_model_fname], True)
            curr_models = self.tfm.getall("wcurr")
            player_names = [player_name_curr, player_name_best]
            player_configs = [player_config_curr, player_config_best]
            cpt = Competition(player_names, player_configs, gamemode = len(player_names), nthreads=self.hpm["nworkers"], ort_info=self.ort_info,filemanager=self.tfm, elo_manager=self.em, stopevent=self.stopevent)
            cpt.run(self.hpm["evo_last_best_games"])
            self.em.update_elo()
            self.tfm.save(self.em, "elo")
            self.em.plot_all_elo_progression(filemanager=self.tfm)
            self.em.plot_elo_progression(player_name_curr, player_config_curr, filemanager=self.tfm, plot_split=True)
            elo_best, ci_best = self.em[player_name_best, player_config_best]
            elo_curr, ci_curr = self.em[player_name_curr, player_config_curr]
            print(f"Elo of best model is {elo_best:.1f} ([{ci_best[0]:.1f}, {ci_best[1]:.1f}])")
            print(f"Elo of current model is {elo_curr:.1f} ([{ci_curr[0]:.1f}, {ci_curr[1]:.1f}])")
            if ci_best[1] < ci_curr[0] or self.always_evolve:
                if self.always_evolve:
                    print(f"Evolving because self.always_evolve is True")
                self.tfm.save(self.nw.state_dict(), "wbest", idd="best")
                new_best_model_fname = self.tfm.getlatest("wbest")
                new_player_name_best, new_player_config_best = self.get_players([new_best_model_fname], True)
                self.em.split_off_player(player_name_curr, player_config_curr, new_player_name_best, new_player_config_best)
                evolved = True
            else:
                evolved = False
        old_best_models = self.tfm.getall("wbest")
        if evolved:
            print_important("Evolved model")
            self.em.plusevolve()
            self.tfm.plusevolve()
            self.hpm.plusevolve()
            if self.verbose:
                print(f"Skipping player with weights filename {old_best_models[-1]} (as it was just saved from the current model)")
            old_best_models = old_best_models[:-1]
        else:
            print("Did not evolve model")
            self.tfm.pluscycle()
            self.hpm.pluscycle()
        nbestmodels = len(old_best_models)
        ndefault = len(DEFAULT_PLAYER_NAMES)
        nmodels = min(ndefault + nbestmodels, self.hpm["max_benchmark_opponents"])
        best_model_names, best_model_configs = self.get_players(old_best_models, True, nosqueeze=True)
        best_model_weights = [self.em.get_weights(name, config) for name, config in zip(best_model_names, best_model_configs)]
        default_model_weights = np.array([self.em.get_weights(name, config) for name, config in zip(DEFAULT_PLAYER_NAMES, DEFAULT_PLAYER_CONFIGS)])
        if best_model_weights:
            default_model_weights *= self.hpm["default_best_bm_ratio"] * sum(best_model_weights) / sum(default_model_weights)
        weights = np.array(list(default_model_weights) + best_model_weights)
        weights = weights / sum(weights)
        chosen_indices = []
        while len(chosen_indices) < nmodels:
            choice = random.choices(range(len(weights)), weights=list(weights), k=1)[0]
            if choice not in chosen_indices:
                chosen_indices.append(choice)
        opp_names = DEFAULT_PLAYER_NAMES + best_model_names
        opp_configs = DEFAULT_PLAYER_CONFIGS + best_model_configs
        print(f"Running benchmark")
        print(f"Choosing opponents from player pool with choice distribution:")
        for name, weight, config in zip(opp_names, weights, opp_configs):
            print(f"  {name}: {weight*100:.1f}%", end='')
            if name == "NNMCTS":
                print(f" (fname: {clip_cfg(config)['fname']})")
            else:
                print()
        chosen_names = [opp_names[i] for i in chosen_indices]
        chosen_configs = [opp_configs[i] for i in chosen_indices]
        player_name_curr, player_config_curr = self.get_players([self.tfm.getlatest("wcurr")], True)
        bm_names = [player_name_curr] + chosen_names
        bm_configs = [player_config_curr] + chosen_configs
        cpt = Competition(bm_names, bm_configs, nthreads=self.hpm["nworkers"], ort_info=self.ort_info, filemanager=self.tfm, elo_manager=self.em, idd=f"benchmark", stopevent=self.stopevent)
        ratio = self.hpm["evo_bm_lb_games_ratio"]
        assert self.nplayers == 2
        nrounds = (len(bm_names) * (len(bm_names) - 1) / 2)
        total_games = ratio * self.hpm["evo_last_best_games"]
        games_per_round =  total_games / nrounds
        games_per_round = int((1 + games_per_round // self.hpm["nworkers"]) * self.hpm["nworkers"])
        print(f"Will be running a total of {int(games_per_round * nrounds)} games; target is {int(total_games)} games")
        cpt.run(games_per_round)

    def model_evolution_benchmark(self):
        print(f"Running final benchmark")
        best_model_fnames = self.tfm.getall("wbest")
        nmodels = len(best_model_fnames)
        if nmodels >= 2:
            print(f"Running competition between NNMCTS players")
            current_model_fname = self.tfm.getlatest("wcurr")
            max_other_opp = self.hpm["max_benchmark_opponents"] - 2
            nother_opp = nmodels - 2
            indices = []
            if min(max_other_opp, nother_opp) > 0:
                ratio = nother_opp / max_other_opp
                indices = [round(ratio * i + .01) for i in range(max_other_opp)]
            indices = [0] + indices + [-1]
            indices = list(set(indices))
            fnames = [best_model_fnames[i] for i in indices]
            fnames += [current_model_fname]
            cpt = Competition(*self.get_players(fnames, True), nthreads=self.hpm["nworkers"], ort_info=self.ort_info, filemanager=self.tfm, idd="NNMCTS_benchmark", stopevent=self.stopevent)
            cpt.run(self.hpm["evo_last_best_games"])
        print(f"Running competition vs. default players")
        fnames = [best_model_fnames[-1]] + DEFAULT_PLAYER_NAMES
        cpt = Competition(*self.get_players(fnames, True), nthreads=self.hpm["nworkers"], ort_info=self.ort_info, filemanager=self.tfm, idd=f"default_player_benchmark", ladder_cpt=True, stopevent=self.stopevent)
        cpt.run(self.hpm["evo_last_best_games"])
 
    def get_players(self, fnames, competition, nosqueeze=False):
        player_names = []
        player_configs = []
        nworkers = self.hpm["nworkers"]
        player_name = "NNMCTS"
        if competition:
            maxit = self.hpm["competition_mcts_maxit"]
        else:
            total_rounds = (self.tfm.cycle_ctr + self.tfm.evo_ctr) * self.hpm["train_play_rounds_per_cycle"] + self.tfm.round_ctr
            it_incr_per_round = max(1, self.hpm["self_play_mcts_maxit"] / (self.hpm["train_play_rounds_per_cycle"] * 2.5))
            linear_maxit = max(1, (total_rounds + 1) * it_incr_per_round)
            maxit = int(min(self.hpm["self_play_mcts_maxit"], linear_maxit))
        player_config = {
            "hyperparams": 
                {
                    "maxit": maxit, 
                    "cpuct": 2, 
                    "epsilon": .25, 
                    "alpha": .2, 
                    "noise": not competition
                }, 
            "ort_info": "", 
            "return_type": "best_action" if competition else "unique_probabilities"
        }
        for fname in fnames:
            if fname in DEFAULT_PLAYER_NAMES:
                player_names.append(fname)
                player_configs.append(DEFAULT_PLAYER_CONFIGS[DEFAULT_PLAYER_NAMES.index(fname)])
            else:
                player_names.append("NNMCTS")
                cfg = player_config
                cfg["fname"] = fname
                player_configs.append(copy.deepcopy(cfg))
        if len(player_names) == 1 and not nosqueeze:
            return player_names[0], player_configs[0]
        return player_names, player_configs

def main(ort_info):
    nworkers = 4 if os.cpu_count() == 8 else 16
    dummy_hyperparams = {
            "min_value_loss_train_weight": .1,
            "replay_buffer_size": 2000, 
            "nworkers": nworkers,
            "self_play_epochs_per_cycle": .15,
            "self_play_mcts_maxit": 50,
            "competition_mcts_maxit": 10,
            "max_cycle_not_evolved": 2,
            "train_play_rounds_per_cycle": 2,
            "evo_last_best_games": nworkers,
            "evo_bm_lb_games_ratio": 1.5,
            "train_batch_size": 8,
            "train_epochs": 1,
            "max_gamma_decay": .1,
            "default_best_bm_ratio":1,
            "lr": 1e-3,
            "play_params": {
                "tau": 1,
                "best_play_move": 10,
            },
            "nplayers": 2,
            "always_evolve": False,
            "max_benchmark_opponents": 1,
    }
    with open("hyperparameters/dummy.pkl", "wb") as f:
        pickle.dump(dummy_hyperparams, f)
    quick_hyperparams = {
            "min_value_loss_train_weight": .2,
            "replay_buffer_size": 150000,
            "nworkers": 4 if os.cpu_count() == 8 else 16,
            "self_play_epochs_per_cycle": .15,
            "self_play_mcts_maxit": 80,
            "competition_mcts_maxit": 100,
            "train_play_rounds_per_cycle": 4,
            "max_cycle_not_evolved": 10,
            "evo_last_best_games": 160,
            "evo_bm_lb_games_ratio": 1,
            "train_batch_size": 512,
            "default_best_bm_ratio":1,
            "train_epochs": 6,
            "max_gamma_decay": .1,
            "lr": 1e-3,
            "play_params": {
                "tau": 1,
                "best_play_move": 10,
            },
            "nplayers": 2,
            "max_benchmark_opponents": 3,
    }
    hyperparams = {
            "min_value_loss_train_weight": .1,
            "replay_buffer_size": 200000,
            "nworkers": 4 if os.cpu_count() == 8 else 16,
            "self_play_epochs_per_cycle": .15,
            "self_play_mcts_maxit": 450,
            "train_play_rounds_per_cycle": 5,
            "competition_mcts_maxit": 700,
            "default_best_bm_ratio":1,
            "max_cycle_not_evolved": 10,
            "evo_last_best_games": 400,
            "evo_bm_lb_games_ratio": 1,
            "train_batch_size": 512,
            "train_epochs": 7,
            "max_gamma_decay": .1,
            "lr": 1e-3,
            "play_params": {
                "tau": 1,
                "best_play_move": 10,
            },
            "nplayers": 2,
            "max_benchmark_opponents": 4,
    }
    nplayers = 2
    old_session_idx = None
    trainer = Trainer(hyperparams=quick_hyperparams, ort_info=ort_info, old_session_idx = old_session_idx, verbose=False)
    trainer.train()

if __name__ == "__main__":
    from mp_ort_import import exec_main
    exec_main(main)
