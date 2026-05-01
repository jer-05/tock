import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import traceback

from fasttock import AVERAGE_GAME_LENGTHS
from data import GameData
from network import PolicyNN
from utils import clear_lines

device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

class NNTrainer:
    def __init__(self, model, data, *, data_ages=None, filemanager=None, datafname=None, verbose=False):
        print(f"Initializing NNTrainer")
        self.nplayers = model.nplayers
        self.verbose = verbose
        self.set_nw(model)
        self.train_iteration = 0
        if datafname is not None:
            self.data = GameData(nplayers=self.nplayers)
            self.data.load_dir(datafname)
        else:
            self.data = data
        self.data_ages = data_ages
        self.maxk = 4
        self.checkpoints = []
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
        self.tfm = filemanager
        if device == torch.device("cuda"):
            print("NNTrainer: GPU is available")
        else:
            print("NNTrainer: GPU not available")

    def set_nw(self, nw):
        if type(nw) is str:
            self.nw = PolicyNN(self.nplayers)
            self.nw.load(nw, silent=not self.verbose)
        elif isinstance(nw, PolicyNN):
            self.nw = nw
        else:
            print(f"NNTrainer.set_nw: Bad network {nw}?")
            breakpoint()
        old_device = next(self.nw.parameters()).device
        self.nw = self.nw.to(device)
        new_device = next(self.nw.parameters()).device
        if self.verbose:
            print(f"Set model was from device {old_device} to {new_device}")

    def get_value_weights(self, move, l, min_value_loss_weight):
        no_loss_move = torch.max(torch.ones(l.shape), .75*l)
        # reversed exponential decay
        value_decay = np.clip(min_value_loss_weight, .01, 1) ** ( 1/ no_loss_move)
        value_weights = 1 + min_value_loss_weight - value_decay ** (no_loss_move - torch.max(torch.zeros(move.shape), no_loss_move - move))
        return value_weights

    def plot_value_weight(self, maxl, min_value_loss_weight):
        fig, ax = plt.subplots()
        fig.suptitle("Weights for value loss")
        ax.set_xlabel("Move")
        ax.set_ylabel("Weight")
        move = list(range(maxl))
        torchmove = torch.Tensor(move)
        for l in range(int(.3*maxl), maxl+1, int(.1*maxl)):
            larr = [l] * maxl
            torchl = torch.Tensor(larr)
            value_weights = self.get_value_weights(torchmove, torchl, min_value_loss_weight)
            npw = value_weights.detach().cpu().numpy()
            ax.plot(move, npw, label=f"l={l}")
        ax.set_ylim((0, 1.05))
        ax.legend(loc="upper right")
        fname = "figures/training/value_weights.png"
        fig.savefig(fname)
        print(f"Saved plot of value weights as {fname}.")

    def train_nn(self, epochs, base_lr, batch_size, *, gamma=None, train_ratio = 0.8, cutoff_lr = 1e-6, min_value_loss_weight=None, fname = None, save_checkpoints=False, reduce_lr_on_plateau=True, keep_progress_text=False, ncheckpoints=20, patience=2):
        lr = base_lr
        steps = round(len(self.data) / batch_size * epochs)
        print("-----------------")
        print("Training PolicyNN")
        print(f"Total amount of samples is {len(self.data)}")
        appr_epochs = batch_size * steps / len(self.data)
        print(f"Will be training NN for {steps} steps with batch size {batch_size} (~= {appr_epochs:.1f} epochs)")
        print("-------------------")
        valid_ratio = 1- train_ratio
        train_set, valid_set = random_split(self.data, [train_ratio, valid_ratio])
        if self.data_ages is None:
            train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        else:
            print(f"Using age decay sampling with gamma={gamma:.3f}")
            ages = self.data_ages[len(self.data_ages) - len(self.data): len(self.data_ages)]
            weights = (gamma ** np.array(ages))
            weights /= max(weights)
            unique_ages = sorted(set(ages))
            unique_weights = gamma ** np.array(unique_ages)
            unique_weights /= max(unique_weights)
            age_counts = [ages.count(unique_age) for unique_age in unique_ages]
            count_pct = np.array(age_counts) / sum(age_counts) * 100
            print("[pct. of samples, sampling weight]: ", end='')
            for i, (weight, agepct) in enumerate(zip(unique_weights, count_pct)):
                print(f"[{agepct:.1f}%, {weight:.3f}]", end='')
                if i == len(count_pct) - 1:
                    print()
                else:
                    print(", ", end='')
            train_idxs = train_set.indices
            valid_idxs = valid_set.indices
            weights_train = list(weights[train_idxs])
            weights_valid = list(weights[valid_idxs])
            sampler_train = WeightedRandomSampler(
                weights=weights_train,
                num_samples = len(train_set),
                replacement=True,
            )
            sampler_valid = WeightedRandomSampler(
                weights=weights_valid,
                num_samples = len(valid_set),
                replacement=True,
            )
            train_dl = DataLoader(train_set, batch_size=batch_size, sampler=sampler_train)
            valid_dl = DataLoader(valid_set, batch_size=batch_size, sampler=sampler_valid)


        optimizer = torch.optim.AdamW(params=self.nw.parameters(), lr=lr, weight_decay = 0.01)
        if reduce_lr_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer,
                mode = 'min',
                factor = 0.1,
                patience = patience,
                threshold = 1e-4,
            )

        checkpoint = max(1, steps // ncheckpoints)
        self.train_losses.append([])
        self.valid_losses.append([])
        self.checkpoints.append([])
        self.train_accs.append([[] for i in range(self.maxk)])
        self.valid_accs.append([[] for i in range(self.maxk)])

        step = 0
        cutoff=False

        policy_loss_func = nn.CrossEntropyLoss()
        if min_value_loss_weight is not None:
            reduction = 'none'
            print(f"Using value loss decay (maximum decay factor is {min_value_loss_weight})")
        else:
            reduction = 'mean'
        if self.nplayers == 2:
            value_loss_func = nn.MSELoss(reduction=reduction)
        else:
            value_loss_func = nn.CrossEntropyLoss(reduction=reduction)

        self.nw.train()
        try:
            while step < steps and not cutoff:
                for batch, (X , (y_prob, y_val, move, l)) in enumerate(train_dl):
                    X = [x.to(device) for x in X]
                    y_prob, y_val = y_prob.to(device), y_val.to(device)
                    logits, value_head = self.nw(*X)
                    policy_loss = policy_loss_func(logits, y_prob)
                    value_loss = value_loss_func(value_head, y_val.view(-1, 1) if self.nplayers == 2 else y_val)
                    if min_value_loss_weight is not None:
                        value_weights = self.get_value_weights(move, l, min_value_loss_weight)
                        if batch == 0:
                            unweighed_value_loss = torch.Tensor(value_loss.shape)
                            unweighed_value_loss.copy_(value_loss)
                            vec_value_loss = torch.Tensor(value_loss.shape)
                            vec_value_loss.copy_(value_loss * value_weights)
                        value_loss = torch.mean(value_loss * value_weights)
                        if batch == 0:
                            self.plot_value_weight(AVERAGE_GAME_LENGTHS[str(self.nplayers)], min_value_loss_weight)
                            shown = 16
                            print(f"Value losses:")
                            print(f" Move, Game Length, Weight, Loss, Weighed Loss")
                            print(f'{"\n".join([f"  {int(move.item())}, {int(l.item())}, {weight.item():.3f}, {loss.item():.3f}, {wloss.item():.3f}" for move, l, weight, loss, wloss in zip(move[:shown], l[:shown], value_weights[:shown], unweighed_value_loss[:shown], vec_value_loss[:shown])])}')
                            print(f"Total loss: {value_loss.item():.3f}")
                            breakpoint()
                    loss = policy_loss + value_loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if step % checkpoint == checkpoint - 1 :
                        train_loss, train_acc = self.check_loss(train_dl, min_value_loss_weight)
                        valid_loss, valid_acc = self.check_loss(valid_dl, min_value_loss_weight)
                        self.train_losses[self.train_iteration].append(train_loss)
                        self.valid_losses[self.train_iteration].append(valid_loss)
                        self.checkpoints[self.train_iteration].append(step)
                        for k in range(self.maxk):
                            self.train_accs[self.train_iteration][k].append(train_acc[k])
                            self.valid_accs[self.train_iteration][k].append(valid_acc[k])
                        if reduce_lr_on_plateau:
                            scheduler.step(sum(valid_loss))
                            lr = scheduler.get_last_lr()[0]
                            if lr < cutoff_lr:
                                print(f"Learning rate is now {lr} < {cutoff_lr} (cutoff), so aborted training.")
                                cutoff=True
                                break
                        
                        train_acc_s = ""
                        valid_acc_s = ""
                        for k in range(self.maxk):
                            train_acc_s += f"k={k+1}: {train_acc[k]*100:.1f}%  "
                            valid_acc_s += f"k={k+1}: {valid_acc[k]*100:.1f}%  "
                        if step != checkpoint - 1 and not keep_progress_text:
                            clear_lines(6)
                        print(f"[{step+1}/{steps}]:\n"
                              f"\tTrain: policy loss {train_loss[0]:.4f}, value loss {train_loss[1]:.4f}{f', weighed value loss {train_loss[2]:.4f}' if min_value_loss_weight else ""}\n\t\tacc. ", train_acc_s,
                              f"\n\tValid: policy loss {valid_loss[0]:.4f}, value loss {valid_loss[1]:.4f}{f', weighed value loss {valid_loss[2]:.4f}' if min_value_loss_weight else ""}\n\t\tacc. ", valid_acc_s,
                              f"\n\tLearning Rate: {lr}"
                        )
                        if save_checkpoints:
                            torch.save(self.nw.state_dict(), f"weights/checkpoints/checkpoint_{(step+1) //checkpoint}.pth")
                    step += 1
                    if step >= steps:
                        break
        except KeyboardInterrupt:
            print(f"[!]: Training was interrupted")
        appr_epochs = step * batch_size / len(self.data)
        print(f"Trained NN for {step} steps (~= {appr_epochs:.1f} epochs)")
        print("-------------------")
        self.train_iteration += 1
        if fname is not None:
            torch.save(self.nw.state_dict(), fname)
        if self.tfm is not None:
            self.tfm.save(self.nw.state_dict(), "wcurr")
        else:
            if fname is None:
                fname = f"weights/{self.nplayers}_players/datasize_{len(self.data)}_epochs_{epochs}_batch_{batch_size}_lr_{base_lr}_iteration_{self.train_iteration}.pth"
            torch.save(self.nw.state_dict(), fname)
            print(f"NNTrainer.train_nn: Saved model as {fname}")
        self.test_current(min_value_loss_weight)
        self.nw = self.nw.to('cpu')
        self.plot_losses_accs()
            
    def get_top_k_acc(self, output, target, k=5):
        with torch.no_grad():
            # Get the index of the best move from the data set
            target_idx = torch.argmax(target, dim=1)
            
            # Get the indices of the top K moves from the NN
            _, pred_indices = output.topk(k, dim=1)
            
            # Check if the target index is anywhere in those top K
            correct = pred_indices.eq(target_idx.view(-1, 1).expand_as(pred_indices))
            return correct.float().sum() / target.size(0)

    def check_loss(self, dl, min_value_loss_weight):
        if min_value_loss_weight is not None:
            reduction = 'none'
        else:
            reduction = 'mean'
        if self.nplayers == 2:
            value_loss_func = nn.MSELoss(reduction=reduction)
        else:
            value_loss_func = nn.CrossEntropyLoss(reduction=reduction)
        policy_loss_func = nn.CrossEntropyLoss()

        self.nw.eval()
        total_loss = np.zeros(2 if min_value_loss_weight is None else 3)
        accs = np.zeros(self.maxk)
        with torch.no_grad():
            for batch, (X , (y_prob, y_val, move, l)) in enumerate(dl):
                X = [x.to(device) for x in X]
                y_prob, y_val = y_prob.to(device), y_val.to(device)
                logits, value_head = self.nw(*X)
                for k in range(self.maxk):
                    accs[k] += self.get_top_k_acc(logits, y_prob, k=k+1)
                policy_loss = policy_loss_func(logits, y_prob)
                # if batch == 0:
                #     print(f"Logits\nTarget probabilities")
                #     print("\n".join([f"{i}: {', '.join([f'{k}: {l:.3f}' for k, l in enumerate(logit) if l > -1000])}\n  {', '.join([f'{j}: {_p:.3f}' for j, _p in enumerate(p) if _p != 0])}" for i, (logit, p) in enumerate(zip(logits, y_prob))]))
                #     print(f"Av. policy loss: {policy_loss.cpu().item() / len(logits):.3f}")
                #     breakpoint()
                value_loss = value_loss_func(value_head, y_val.view(-1, 1) if self.nplayers == 2 else y_val)
                if min_value_loss_weight is not None:
                    no_loss_move = l - .25 * AVERAGE_GAME_LENGTHS[str(self.nplayers)]
                    value_decay = min_value_loss_weight ** ( 1/ no_loss_move)
                    value_weights = value_decay ** torch.max(torch.zeros(move.shape), no_loss_move - move)
                    weighed_value_loss = torch.mean(value_loss * value_weights)
                    value_loss = torch.mean(value_loss)
                loss = [policy_loss.cpu().item(), value_loss.cpu().item()]
                if min_value_loss_weight:
                    loss.append(weighed_value_loss.cpu().item())
                total_loss += np.array(loss)

        av_loss = total_loss / len(dl)
        accs = accs / len(dl)
        self.nw.train()
        return av_loss, accs

    def test_current(self, min_value_loss_weight):
        print("----------------")
        print(f"Testing performance of model on dataset (length {len(self.data)})")
        dl = DataLoader(self.data, batch_size=64)
        av_loss, accs = self.check_loss(dl, min_value_loss_weight)
        print(f"Loss: policy -> {av_loss[0]:.3f}, value -> {av_loss[1]:.3f}", end="")
        if len(av_loss) == 3:
            print(f", weighed value -> {av_loss[2]:.3f}")
        else:
            print()
        acc_s = ""
        for k in range(self.maxk):
            acc_s += f"k={k+1}: {accs[k]*100:.1f}%  "
        print(f"Acc.: ", acc_s)
        print("-----------------")
        
    def plot_losses_accs(self, plotall=False, fname=None):
        try:
            print("Plotting training progress")
            if not plotall:
                train_losses = self.train_losses[-1]
                valid_losses = self.valid_losses[-1]
                train_accs = self.train_accs[-1]
                valid_accs = self.valid_accs[-1]
                checkpoints = self.checkpoints[-1]
            else:
                train_losses = self.train_losses[0]
                valid_losses = self.valid_losses[0]
                train_accs = self.train_accs[0]
                valid_accs = self.valid_accs[0]
                checkpoints = self.checkpoints[0]
                for i in range(1, len(self.train_losses)):
                    train_losses += self.train_losses[i]
                    valid_losses += self.valid_losses[i]
                    for k in range(self.maxk):
                        train_accs[k] += self.train_accs[i][k]
                        valid_accs[k] += self.valid_accs[i][k]
                    last_checkpoint = checkpoints[-1]
                    checkpoints += [self.checkpoints[i][j] + last_checkpoint for j in range(len(self.checkpoints[i]))]
            train_losses = np.array(train_losses)
            valid_losses = np.array(valid_losses)
            train_accs = np.array(train_accs)
            valid_accs = np.array(valid_accs)
            checkpoints = np.array(checkpoints)

            fig, ax = plt.subplots(len(train_losses[0]) + 2, figsize = (9, 9) if not plotall else (18, 9))
            fig.suptitle("Training progress", fontsize=18)
            ax_idx = 0
            ax[ax_idx].set_title("Policy Loss")
            ax[ax_idx].plot(checkpoints, train_losses[:, 0], label="Train Loss")
            ax[ax_idx].plot(checkpoints, valid_losses[:, 0], label="Validation Loss")
            ax[ax_idx].set_ylabel("Normalized Loss")
            y_lim = ax[ax_idx].get_ylim()
            maxy = y_lim[1]
            ax[ax_idx].set_ylim((0, max(1, maxy)))
            ax[ax_idx].legend(bbox_to_anchor=(1, 1.02), loc='lower right')
            ax_idx += 1

            ax[ax_idx].set_title("Value Loss")
            ax[ax_idx].plot(checkpoints, train_losses[:, 1], label="Train Loss")
            ax[ax_idx].plot(checkpoints, valid_losses[:, 1], label="Validation Loss")
            ax[ax_idx].set_ylabel("Normalized Loss")
            y_lim = ax[ax_idx].get_ylim()
            maxy = y_lim[1]
            ax[ax_idx].set_ylim((0, max(1, maxy)))
            ax_idx += 1
            if len(train_losses[0]) == 3:
                ax[ax_idx].set_title("Weighed Value Loss")
                ax[ax_idx].plot(checkpoints, train_losses[:, 2], label="Train Loss")
                ax[ax_idx].plot(checkpoints, valid_losses[:, 2], label="Validation Loss")
                ax[ax_idx].set_ylabel("Normalized Loss")
                y_lim = ax[ax_idx].get_ylim()
                maxy = y_lim[1]
                ax[ax_idx].set_ylim((0, max(1, maxy)))
                ax_idx += 1
            
            ax[ax_idx].set_title("Training Accuracy")
            for k in range(len(train_accs)):
                ax[ax_idx].plot(checkpoints, train_accs[k] * 100, label=f"k={k+1}")
            ax[ax_idx].legend(loc='lower right', fontsize=8)
            ylim = ax[ax_idx].get_ylim()
            ax[ax_idx].set_ylim((ylim[0]-20, 100))
            ax[ax_idx].set_ylabel("Accuracy [%]")
            ax_idx += 1

            ax[ax_idx].set_title("Validation Accuracy")
            for k in range(len(valid_accs)):
                ax[ax_idx].plot(checkpoints, valid_accs[k] * 100, label=f"k={k+1}")
            ax[ax_idx].set_ylim((ylim[0]-20, 100))
            ax[ax_idx].set_ylabel("Accuracy [%]")
            ax[ax_idx].set_xlabel("Gradient descent steps")

            fig.tight_layout()

            if self.tfm is not None:
                self.tfm.save(fig, "train", idd="all" if plotall else None)
            else:
                if fname is None:
                    fname = f"figures/training/training_losses_accs_{'all_' if plotall else ''}nplayers_{self.nplayers}_datasize_{len(self.data)}_iteration_{self.train_iteration}.png"
                print(f"NNTrainer.plot_losses_accs: Saved {'all' if plotall else 'last'} training losses and accuracies as {fname}")
                fig.savefig(fname)
            plt.close('all')
        except Exception:
            traceback.print_exc()
            breakpoint()

def main():
    model = PolicyNN(nplayers=2)
    # datafname ="data/2_players/TSPEnv_maxdepth-5_return_type-best_action_size_1529"
    datafname = 'data/2_players/TSPEnv_maxdepth-5_return_type-best_action_size_59981'

    # datafname = "/home/jer/Projects/machine_learning/keezenspel/data/2_players/TSPEnv_maxdepth-5_TSPEnv_maxdepth-5_size_300005"
    nn_trainer = NNTrainer(model, None, datafname=datafname)
    # nn_trainer.train_nn(10, 1e-3, 256, keep_progress_text=True, ncheckpoints=30, min_value_loss_weight=.1, patience=2)
    # nn_trainer.train_nn(1, 1e-3, 256, keep_progress_text=True, ncheckpoints=4, min_value_loss_weight=.1, patience=2)
    nn_trainer.plot_value_weight(150, 0)
    breakpoint()
    # nn_trainer.train_nn(5, 1e-3, 128, keep_progress_text=True, ncheckpoints=10, min_value_loss_weight=.1)
    # nn_trainer.plot_losses_accs(plotall=True)

if __name__ == '__main__':
    main()

