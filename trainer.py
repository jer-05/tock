import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import os
import traceback
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from tock import make, DEAL_ORDER, AVERAGE_GAME_LENGTH
from network import PolicyNN
from utils import policy_to_actions, remove_dirs, clear_lines
from data import GameData
from players import NNPlayer
from mcts import MCTS, Node, totorch
from competition import Competition
from datagen import DataGenerator
from gpu_manager import get_gpu_manager
from filemanager import TrainingFileManager
from hyperparameter_manager import HyperparameterManager

device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

WEIGHTS_DIR = "weights"
TRAINING_FIG_DIR = "figures/training"

def get_value_weights(move_values, reference_move=AVERAGE_GAME_LENGTH, nplayers=2):
    sigmoid_halfpoint = reference_move // 2 
    smoothening = reference_move // 8
    sigmoids = 1 / (1 + torch.exp((move_values - sigmoid_halfpoint) / smoothening))
    return torch.clip(sigmoids, 0.1, 1)

class Trainer():
    def __init__(self, nn, *, data_fname=None, weights_fname=None, hyperparams={}, loadblob=False):
        self.datalen_ctr = 0
        self.nw = nn.to(device)

        self.tfm = TrainingFileManager()
        self.hpm = HyperparameterManager(hyperparams)

        self.data = GameData()
        self.pretraining = False
        if data_fname is not None:
            print(f"Loading pre-generated data from {data_fname}")
            if loadblob:
                self.data.load_blobs(data_fname)
            else:
                self.data.load(data_fname)
            self.pretraining = True
        else:
            self.pretraining = False
        self.data.cut_buffer_size(self.hpm["replay_buffer_size"])

        if weights_fname is not None:
            print(f"Loading pre-trained model from {weights_fname}")
            self.nw.load(weights_fname)
        else:
            print(f"Starting from scratch; no pre-trained model was loaded")
        self.tfm.save(self.nw.state_dict(), "wcurr")

        if device == torch.device("cuda"):
            print("Training on GPU")
        else:
            print("Training on CPU")
        self.cpt_log = "competition_log.txt"
        self.train_log = "train_log.txt"
        self.datagen_log = "datagen_log.txt"

    def gen_train_data(self, startup):
        if not startup:
            maxit = self.hpm["self_play_mcts_maxit"]
        else:
            maxit = 50
            print(f"Using startup maxit of {maxit}!")
        self.data.cut_buffer_size(self.hpm["replay_buffer_size"])

        print(f"Starting self-play")
        former_len = len(self.data)
        print(f"Already {former_len} samples in database")
        if self.hpm.switch_datagen_model(verbose=True):
            model_path = self.tfm.getlatest("wbest")
        else:
            model_path = self.tfm.getlatest("wcurr")
        gpu_mgr, player_names, player_configs =\
                self.get_gpu_mgr_and_players(model_path, maxit, return_type="probabilities", get_n_players=True)
        generator = DataGenerator(player_names, player_configs, self.hpm["play_params"], nthreads=self.hpm["nworkers"], gpu_manager=gpu_mgr)
        data = generator.generate(self.hpm["self_play_ngames"], return_data=True)
        self.datalen_ctr += len(data)
        self.data.load(data=data, load_from_data=True, append=True)
        if self.datalen_ctr > self.hpm["replay_buffer_size"]:
            self.tfm.save(self.data, "data")
            self.datalen_ctr = 0

    def train_nn(self, *, epochs=None, train_ratio = 0.8, lr = 1e-3, gamma = .98, cutoff_lr = 1e-6, use_valid_set = True, use_value_scaling=False, fname = None, save_checkpoints=False, reduce_lr_on_plateau=False):
        batch_size = self.hpm["train_batch_size"]
        if epochs:
            steps = round(len(self.data) / batch_size * epochs)
        else:
            steps = round(self.hpm["train_steps"])
        print("-----------------")
        print("Training PolicyNN")
        print(f"Total amount of samples is {len(self.data)}")
        appr_epochs = batch_size * steps / len(self.data)
        print(f"Will be training NN for {steps} steps (~= {appr_epochs:.1f} epochs)")
        print("-------------------")
        if use_valid_set:
            valid_ratio = 1- train_ratio
            train_set, valid_set = random_split(self.data, [train_ratio, valid_ratio])
            train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        else:
            train_dl = valid_dl = DataLoader(self.data, batch_size=batch_size, shuffle = True)

        optimizer = torch.optim.AdamW(params=self.nw.parameters(), lr=lr, weight_decay = 0.01)
        if reduce_lr_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer,
                mode = 'min',
                factor = 0.1,
                patience = 1,
                threshold = 1e-4,
            )
        maxk = 4

        checkpoint = max(1, steps // 20)
        train_losses = []
        valid_losses = []
        train_accs = [[] for i in range(maxk)]
        valid_accs = [[] for i in range(maxk)]

        step = 0
        cutoff=False

        try:
            while step < steps and not cutoff:
                self.nw.train()
                try:
                    for batch, (X , (y_prob, y_val)) in enumerate(train_dl):
                        # print(f"Now doing batch {batch}!")
                        X = [x.to(device) for x in X]
                        y_prob, y_val = y_prob.to(device), y_val.to(device)
                        # print(f"About to pass to nn!")
                        # print(self.nw)
                        logits, value_head = self.nw(*X)
                        # print(f"Calculating loss!")
                        policy_loss = self.nw.p_loss(logits, y_prob)

                        # reduce gradients in opening. Factor in front of X[5] (rounds) is an estimation for the average amount of moves per round
                        if use_value_scaling:
                            value_weights = 1 - get_value_weights(X[5].detach() * self.hpm["nplayers"] / 2 * sum(DEAL_ORDER) / len(DEAL_ORDER))
                            value_loss = self.nw.v_loss(value_head * value_weights, y_val.view(-1, 1) * value_weights)
                        else:
                            value_loss = self.nw.v_loss(value_head, y_val.view(-1, 1) if self.hpm["nplayers"] == 2 else y_val)
                        loss = policy_loss + value_loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        if step % checkpoint == checkpoint - 1 :
                            train_loss, train_acc = self.check_loss(train_dl)
                            valid_loss, valid_acc = self.check_loss(valid_dl)
                            train_losses.append(train_loss)
                            valid_losses.append(valid_loss)
                            for k in range(maxk):
                                train_accs[k].append(train_acc[k])
                                valid_accs[k].append(valid_acc[k])
                            if reduce_lr_on_plateau:
                                scheduler.step(sum(valid_loss))
                                lr = scheduler.get_last_lr()[0]
                                if lr < cutoff_lr:
                                    print(f"Learning rate is now {lr} < {cutoff_lr} (cutoff), so aborted training.")
                                    cutoff=True
                                    break
                            
                            train_acc_s = ""
                            valid_acc_s = ""
                            for k in range(maxk):
                                train_acc_s += f"k={k+1}: {train_acc[k]*100:.1f}%  "
                                valid_acc_s += f"k={k+1}: {valid_acc[k]*100:.1f}%  "
                            if step != checkpoint - 1:
                                clear_lines(6)
                            print(f"[{step+1}/{steps}]:\n"
                                  f"\tTrain: policy loss {train_loss[0]:.4f}, value loss {train_loss[1]:.4f}\n\t\tacc. ", train_acc_s,
                                  f"\n\tValid: policy loss {valid_loss[0]:.4f}, value loss {valid_loss[1]:.4f}\n\t\tacc. ", valid_acc_s,
                                  f"\n\tLearning Rate: {lr}"
                            )
                            if save_checkpoints:
                                torch.save(self.nw.state_dict(), f"weights/checkpoints/checkpoint_{(step+1) //checkpoint}.pth")
                        step += 1
                        if step >= steps:
                            break
                except Exception as e:
                    traceback.print_exc()
                    breakpoint()
            appr_epochs = step * batch_size / len(self.data)
            print(f"Trained NN for {step} steps (~= {appr_epochs:.1f} epochs)")
            print("-------------------")
            self.tfm.save(self.nw.state_dict(), "wcurr")
            if fname is not None:
                torch.save(self.nw.state_dict(), fname)
            self.plot_losses_accs(range(len(train_losses)), np.array(train_losses), np.array(valid_losses),\
                    train_accs, valid_accs)
            self.test_current()
        except KeyboardInterrupt:
            self.emergency_save()
            
    def get_top_k_acc(self, output, target, k=5):
        """
        output: [batch, 52] (Raw Logits)
        target: [batch, 52] (Probabilities from Tree Search)
        """
        with torch.no_grad():
            # Get the index of the best move from the Tree Search
            target_idx = torch.argmax(target, dim=1)
            
            # Get the indices of the top K moves from the NN
            _, pred_indices = output.topk(k, dim=1)
            
            # Check if the target index is anywhere in those top K
            # expand_as handles the batch dimension comparison
            correct = pred_indices.eq(target_idx.view(-1, 1).expand_as(pred_indices))
            return correct.float().sum() / target.size(0)

    def check_loss(self, dl, maxk = 4, extra_info=False):
        self.nw.eval()
        total_loss = np.zeros(2)
        accs = np.zeros(maxk)
        with torch.no_grad():
            for batch, (X , (y_prob, y_val)) in enumerate(dl):
                X = [x.to(device) for x in X]
                y_prob, y_val = y_prob.to(device), y_val.to(device)
                logits, value_head = self.nw(*X)
                for k in range(maxk):
                    accs[k] += self.get_top_k_acc(logits, y_prob, k=k+1)
                policy_loss = self.nw.p_loss(logits, y_prob)
                policy = self.nw.softmax(logits)
                value_loss = self.nw.v_loss(value_head, y_val.view(-1, 1) if self.hpm["nplayers"] == 2 else y_val)
                loss = np.array([policy_loss.cpu().item(), value_loss.cpu().item()])
                total_loss += loss
        av_loss = total_loss / len(dl)
        accs = accs / len(dl)
        return av_loss, accs

    def test_current(self, maxk=4):
        print("----------------")
        print(f"Testing performance of model on dataset (length {len(self.data)})")
        dl = DataLoader(self.data, batch_size=64)
        av_loss, accs = self.check_loss(dl, maxk=maxk)
        print(f"Loss: policy -> {av_loss[0]:.2f}, value -> {av_loss[1]:.2f}")
        acc_s = ""
        for k in range(maxk):
            acc_s += f"k={k+1}: {accs[k]*100:.1f}%  "
        print(f"Acc.: ", acc_s)
        print("-----------------")
        
    def plot_losses_accs(self, checkpoints, train_losses, valid_losses, train_accs, valid_accs):
        print("Plotting training progress")
        fig, ax = plt.subplots(4, figsize = (9, 9))
        fig.suptitle("Training progress")
        ax[0].set_title("Policy Loss")
        ax[0].plot(checkpoints, train_losses[:, 0], label="Train Loss")
        ax[0].plot(checkpoints, valid_losses[:, 0], label="Validation Loss")
        ax[0].legend()

        ax[1].set_title("Value Loss")
        ax[1].plot(checkpoints, train_losses[:, 1], label="Train Loss")
        ax[1].plot(checkpoints, valid_losses[:, 1], label="Validation Loss")
        ax[1].legend()
        
        ax[2].set_title("Training Accuracy")
        for k in range(len(train_accs)):
            ax[2].plot(checkpoints, train_accs[k], label=f"k={k+1}")
        ax[2].legend()

        ax[3].set_title("Validation Accuracy")
        for k in range(len(valid_accs)):
            ax[3].plot(checkpoints, valid_accs[k], label=f"k={k+1}")
        ax[3].legend()

        fig.tight_layout()
        self.tfm.save(fig, "train")
        plt.close('all')

    def train(self):
        try:
            print("Starting training")

            max_cycles = self.hpm["train_play_cycles"]
            if self.pretraining:
                print(f"Starting training with generated data (already {len(self.data)} samples available)")
                self.train_nn(epochs=50)
            evo_cycle = 0
            while not self.hpm.stop_training():
                print(f"-------------------------------------------")
                print(f"Now starting train and play cycle {evo_cycle}")
                print(self.hpm)
                for cycle in range(max_cycles):
                    print(f"-------------------------------------------")
                    print(f"Now starting train and play round {evo_cycle * max_cycles + cycle}")
                    if evo_cycle == 0:
                        startup = True
                    else:
                        startup = False
                    self.gen_train_data(startup=startup)
                    self.train_nn(lr=self.hpm["lr"])
                print(f"\nChecking improvement of model!")
                self.evolve()
                evo_cycle += 1
            self.model_evolution_benchmark()
        except KeyboardInterrupt:
            self.model_evolution_benchmark()

    def evolve(self):
        if not self.tfm.getlatest("wbest", soft_fail=True):
            print("No best model yet, setting current to best by default")
            self.tfm.save(self.nw.state_dict(), "wbest")
            evolved = True
        else:
            print("Comparing latest model to best")
            best_model_fname = self.tfm.getlatest("wbest")
            current_model_fname = self.tfm.getlatest("wcurr")
            gpu_manager_current, player_name_curr, player_config_curr =\
                    self.get_gpu_mgr_and_players(current_model_fname, self.hpm["competition_mcts_maxit"])
            gpu_manager_best, player_name_best, player_config_best =\
                    self.get_gpu_mgr_and_players(current_model_fname, self.hpm["competition_mcts_maxit"])
            player_names = [player_name_curr, player_name_best]
            player_configs = [player_config_curr, player_config_best]


            cpt = Competition(player_names, player_configs, gpu_managers=[gpu_manager_current, gpu_manager_best], nthreads=self.hpm["nworkers"])
            results, fig = cpt.run(self.hpm["evo_competition_games"], return_results_and_fig=True)
            wins_ratio = results[1, 0]
            self.tfm.save(fig, "cpt")
            fig.clf()
            if wins_ratio > self.hpm["evo_win_criterion"]:
                print(f"Evolving model! (win rate > {self.hpm["evo_win_criterion"]}%)")
                self.tfm.save(self.nw.state_dict(), "wbest")
                evolved = True
            else:
                print(f"Not evolving model (win rate < {self.hpm["evo_win_criterion"]}%)")
                self.hpm.plusnotevolved()
                evolved = False
        if evolved:
            self.hpm.evolved()
            print(f"Best model changed, so running benchmark")
            if self.hpm["evo_benchmark_games"] == 0:
                print(f"Skipping benchmark because ngames=0!")
                return
            current_model_fname = self.tfm.getlatest("wcurr")
            gpu_manager_current, player_name, player_config =\
                    self.get_gpu_mgr_and_players(current_model_fname, self.hpm["competition_mcts_maxit"])
            opp_names = [
                    "NN",
                    "TSPDet",
                    "Random",
            ]
            opp_configs = [
                    {"fname": current_model_fname},
                    {"maxdepth": 6, "maxit": 40},
                    {},
            ]
            for opp_name, opp_config in zip(opp_names, opp_configs):
                player_names = [player_name, opp_name]
                player_configs = [player_config, opp_config]
                cpt = Competition(player_names, player_configs, gpu_managers=[gpu_manager_current], nthreads=self.hpm["nworkers"])
                results, fig = cpt.run(self.hpm["evo_benchmark_games"], return_results_and_fig=True)
                self.tfm.save(fig, "cpt", idd=f"benchmark{opp_name}")

    def model_evolution_benchmark(self):
        print(f"Running final benchmark to compare model generations")
        gpu_mgrs = []
        player_names = []
        player_configs = []
        fnames = self.tfm.getall("wbest")
        if len(fnames) < 2:
            print(f"Only {len(fnames)} model was saved, so cannot run benchmark")
            return
        max_models = 5
        nmodels = len(fnames)
        skip_models = max(0, nmodels - max_models)
        if skip_models > 0:
            skip_frequency = skip_models / nmodels
            play_model = []
            incr = 0
            for i in range(nmodels):
                incr += skip_frequency
                if incr > .5:
                    play_model.append(False)
                    incr -= 1
                else:
                    play_model.append(True)
        else:
            play_model = [True] * nmodels

        print(f"Will be benchmarking models:")
        for i, (play, fname) in enumerate(zip(play_model, fnames)):
            if play:
                print(f"   model {i}: {fname}")
        print(f"(Skipping {skip_models} models)")


        for play, fname in zip(play_model, fnames):
            if not play:
                continue
            gpu_mgr, player_name, player_config =\
                self.get_gpu_mgr_and_players(fname, self.hpm["competition_mcts_maxit"])
            gpu_mgrs += [gpu_mgr]
            player_names += [player_name]
            player_configs += [player_config]
        cpt = Competition(player_names, player_configs, gpu_managers=gpu_mgrs, nthreads=self.hpm["nworkers"])
        results, fig = cpt.run(self.hpm["evo_competition_games"], return_results_and_fig=True)
        self.tfm.save(fig, "cpt", idd="final_benchmark")

 
    def get_gpu_mgr_and_players(self, fname, maxit, return_type=None, get_n_players=False):
        worker_batch_size = self.hpm["worker_batch_size"]
        nworkers = self.hpm["nworkers"]
        gpu_manager, gpu_manager_info = get_gpu_manager(nworkers=nworkers, worker_batch_size=worker_batch_size, model_path=fname)
        player_name = "VNNMCTS"
        player_config = {"hyperparams": {"maxit": maxit, "num_sims": worker_batch_size, "virtual_loss":1}, 'gpu_manager_info': gpu_manager_info}
        if return_type is not None:
            player_config["return_type"] = return_type
        if get_n_players:
            return gpu_manager, [player_name] * self.hpm["nplayers"], [player_config] * self.hpm["nplayers"]
        return gpu_manager, player_name, player_config

    def emergency_save(self):
        print("Emergency save!")
        self.tfm.save(self.nw.state_dict(), "weight", fname="emergency_save.pth")

def main():
    dummy_hyperparams = {
            "replay_buffer_size": 3000, 
            "nworkers": 16,
            "worker_batch_size": 32,
            "self_play_ngames": 16,
            "self_play_mcts_maxit": 1,
            "competition_mcts_maxit": 1,
            "max_cycle_not_evolved": 5,
            "train_play_cycles": 2,
            "evo_competition_games": 16,
            "evo_benchmark_games": 0,
            "evo_win_criterion": 55,
            "train_batch_size": 128,
            "train_steps": 10,
            "lr": 1e-3,
            "play_params": {
                "tau": 1,
                "best_play_move": 10,
            },
            "nplayers": 2,
    }
    hyperparams = {
            "replay_buffer_size": 40000,
            "nworkers": 16,
            "worker_batch_size": 16,
            "self_play_ngames": 100,
            "self_play_mcts_maxit": 500,
            "competition_mcts_maxit": 800,
            "train_play_cycles": 5,
            "max_cycle_not_evolved": 14,
            "evo_competition_games": 500,
            "evo_benchmark_games": 200,
            "evo_win_criterion": 55,
            "train_batch_size": 512,
            "train_steps": 100,
            "lr": 1e-3,
            "play_params": {
                "tau": 1,
                "best_play_move": 10,
            },
            "nplayers": 2,
    }
    nplayers = 2
    trainer = Trainer(PolicyNN(nplayers=nplayers), hyperparams=hyperparams)
    trainer.train()
    # for i in range(7):
    #     trainer.tfm.save(trainer.nw.state_dict(), "wbest")
    #     trainer.tfm.plusevo()
    # trainer.model_evolution_benchmark()

if __name__ == '__main__':
    main()
