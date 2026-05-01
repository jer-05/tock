import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import traceback
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.signal as sl
from itertools import cycle

from tock import RANKS, SUITS, DEAL_ORDER
from fasttock import print_action_space, CARD_TO_REPR, is_capture
from utils import field_start_base_to_obs, totorch, checkfn

class GameData(Dataset):
    """
    X: tuple (field, base, start, card, fold, roundn, player)
    y: tuple (y_prob, y_val)
    """
    def __init__(self, nplayers, buffer_size = 10000000):
        super().__init__()
        self.X = []
        self.y = []
        self.buffer_size = buffer_size
        self.nplayers = nplayers
        self.players = []

    def __getitem__(self, idx, gettorch=True):
        assert len(self) <= self.buffer_size
        sparse_field, start, base, card, fold, roundn, player, sparse_mask = self.X[idx]
        nplayers = len(start) - 3
        assert nplayers == self.nplayers
        field = np.zeros((nplayers + 3, nplayers * 16))
        field[sparse_field] = 1
        action_mask = np.zeros(50, dtype=bool)
        action_mask[sparse_mask] = True
        sparse_prob_idx, sparse_prob, y_val, move, l, playernames = self.y[idx]
        if nplayers != 2:
            tmp = y_val
            y_val = np.zeros(nplayers)
            y_val[tmp] = 1
        dtype = torch.float32
        y_prob = np.zeros(50, dtype=float)
        y_prob[sparse_prob_idx] = sparse_prob
        if gettorch:
            return totorch((field, start, base, card, fold, roundn, player, action_mask)), totorch((y_prob, y_val, move, l))
        return (field, start, base, card, fold, roundn, player, action_mask), (y_prob, y_val, move, l)

    def append(self, field, start, base, card, fold, roundn, player, action_mask, y_prob, y_val, move, game_length, playernames):
        nplayers = len(field) - 3
        if nplayers != self.nplayers:
            print(f"GameData.append: Appending data for a {nplayers} game, but the database is for {self.nplayers} players.")
            breakpoint()
        if abs(sum(y_prob[action_mask.astype('bool')]) - 1) > 1e-3:
            print(f"GameData.append: invalid y_prob i.c.w. action mask, probability sum is not 1")
            print(f"Given action space is:")
            print_action_space(action_mask)
            prob_mask = np.zeros(50, dtype=bool)
            prob_mask[np.nonzero(y_prob)[0]] = True
            print(f"Non-zero y_prob entries correspond to space:")
            print_action_space(prob_mask)
            print(f"Trying to print action space with probabilities")
            print_action_space(action_mask, action_probabilities=y_prob, check_normalization=False)
            breakpoint()
        if abs(sum(y_prob[np.logical_not(action_mask.astype('bool'))])) > 1e-3:
            print(f"GameData.append: invalid y_prob of {y_prob} i.c.w. action mask of {action_mask}, invalid moves have nonzero probability")
            print(f"Given action space is:")
            print_action_space(action_mask)
            prob_mask = np.zeros(50, dtype=bool)
            prob_mask[np.nonzero(y_prob)[0]] = True
            print(f"Non-zero y_prob entries correspond to space:")
            print_action_space(prob_mask)
            breakpoint()
        if len(self.X) >= self.buffer_size:
            self.X.pop(0)
            self.y.pop(0)
        sparse_field = field.nonzero()
        sparse_field = tuple(sparse_field[i].astype('uint8') for i in range(2))
        sparse_mask = action_mask.nonzero()[0].astype('uint8')
        sparse_prob_idx = y_prob.nonzero()[0].astype('uint8')
        sparse_prob = y_prob[sparse_prob_idx].astype('float32')
        # print(y_prob)
        # print(sparse_prob_idx)
        # print(sparse_prob)
        if nplayers != 2:
            y_val = list(y_val).index(1)
        self.X.append((sparse_field, start, base, card, fold, roundn, player, sparse_mask))
        self.y.append((sparse_prob_idx, sparse_prob, y_val, move, game_length, playernames))

    def print(self, idx):
        print(f"--------------------------")
        print(f"Printing data sample {idx}")
        sparse_field, start, base, card, fold, roundn, player, sparse_mask = self.X[idx]
        nplayers = len(start) - 3
        field = np.zeros((nplayers + 3, nplayers * 16), dtype='uint8')
        field[sparse_field] = 1
        action_mask = np.zeros(50, dtype='bool')
        action_mask[sparse_mask] = True
        sparse_prob_idx, sparse_prob, y_val, move, game_length, playernames = self.y[idx]
        if nplayers != 2:
            tmp = y_val
            y_val = np.zeros(nplayers)
            y_val[tmp] = 1
        dtype = torch.float32
        y_prob = np.zeros(50, dtype=float)
        y_prob[sparse_prob_idx] = sparse_prob
        pawn_locs = field_start_base_to_obs(field, start, base)
        print(f"{'Player':<15}: {playernames[0]} (index {player})")
        print(f"{'Opponents':<15}: {', '.join([name for name in playernames[1:]])}")
        print(f"{'Round':<15}: {roundn}")
        print(f"{'Move':<15}: {move}/{game_length}")
        if nplayers != 2:
            y_val = list(y_val)
            y_val_str = ""
            for val in y_val:
                y_val_str += f"{val:.3f}, "
            y_val_str = y_val_str[:-2]
            print(f"Value Target: {y_val_str}")
        else:
            print(f"Value Target: {y_val:.3f}")
        print(f"Pawn Locations:\n")
        for i in range(nplayers):
            if i ==0:
              print(f"    Current player: ", end='')
            else:
              print(f"    Oponent {i}     : ", end='')
            print(f"[{', '.join(format(loc, '>2.0f') for loc in pawn_locs[i])}]")
        card_str = ""
        for i in range(13):
            ncard = int(card[i])
            for j in range(ncard):
                card_str += CARD_TO_REPR[i] + ", "
        card_str = card_str[:-2]
        ncards = int(sum(card))
        print(f"{'Cards':<15}: {card_str}  (total {ncards})")
        print_action_space(action_mask, action_probabilities=y_prob, check_normalization=True, print_delimiters=False, capture_info=[field, base, start])
        assert fold in [0, 1, 3]
        if fold == 0:
            print(f"Previous player folded")
        elif fold == 3:
            print(f"New round started")
        elif fold == 1:
            print(f"Previous player played a card")
        print(f"--------------------------")

    def save(self, dirname, chunk_size = 200000,*, verbose=False):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            print(f"DataGenerator.save: Directory {dirname} exists")
            breakpoint()
        except FileNotFoundError:
            print(f"DataGenerator.save: Parent directory of directory {dirname} does not exist")
            breakpoint()
        assert len(self) > 0, "Cannot save dataset from empty GameData object"
        l = len(self)

        sample = 0
        blobid = 0
        while sample < len(self):
            start = sample
            stop = min(len(self), sample + chunk_size)

            state = {
                    'X': self.X[start:stop],
                    'y': self.y[start:stop],
                    'buffer_size': self.buffer_size,
            }
            blob_fname = f"{dirname}/blob{blobid}.pth"
            if verbose:
                print(f"Saving blob {blobid} as {blob_fname} with data [{start}:{stop}]")
            torch.save(state, blob_fname)
            blobid += 1
            sample += chunk_size
        print(f"GameData: Saved data ({blobid} blobs) to folder '{dirname}'")

    def load_dir(self,dirname, preserve_buffer_size=False, verbose=False):
        i=0
        if not os.path.isdir(dirname):
            print(f"DataGenerator.load_dir: Directory {dirname} does not exist.")
            breakpoint()
        old_size = len(self)
        old_buf_size = self.buffer_size
        while True:
            blobfname =f"{dirname}/blob{i}.pth"
            if not os.path.isfile(blobfname):
                break
            blobdata = torch.load(blobfname, weights_only=False)
            npl = len(blobdata['X'][1][1]) - 3
            if npl != self.nplayers:
                print(f"GameData.load_dir: trying to load data for {npl} players, but this data object is for {self.nplayers} player games")
                breakpoint()
            self.X += blobdata['X']
            self.y += blobdata['y']
            if verbose:
                print(f"Loaded blob {i} with dirname {blobfname} with {len(blobdata['X'])} samples")
            i += 1
        if len(self) > self.buffer_size:
            if not preserve_buffer_size:
                self.set_buffer_size(len(self))
            else:
                self.set_buffer_size(self.buffer_size)
        print(f"GameData: Loaded data from directory {dirname} ({i} blobs). Number of data samples increased from {old_size} to {len(self)}")


    def set_buffer_size(self, size):
        old_size = self.buffer_size
        self.buffer_size = size
        old_len = len(self)
        if old_len > size:
            cutsize = old_len - size
            self.X = self.X[old_len - size:]
            self.y = self.y[old_len - size:]
            print(f"GameData: Set buffer size to {self.buffer_size} (was {old_size}). Cut {cutsize} data samples.")
        else:
            print(f"GameData: Set buffer size to {self.buffer_size} (was {old_size})")

    def load_data(self, data, preserve_buffer_size=True):
        assert data.nplayers == self.nplayers
        old_size = len(self)
        state = copy.deepcopy(data)
        self.X += state.X
        self.y += state.y
        if len(self) > self.buffer_size:
            if preserve_buffer_size:
                self.set_buffer_size(self.buffer_size)
            else:
                self.set_buffer_size(len(self))
        print(f"GameData: Appended data from data object with {len(data)} samples to current data; size changed from {old_size} to {len(self)} samples")


    def show_random(self, n=10, capture_only=False):
        nskipped = 0
        i = 0
        while i < n:
            idx = random.randint(0, len(self) - 1)
            skip=False
            if capture_only:
                sparse_field, start, base, card, fold, roundn, player, sparse_mask = self.X[idx]
                nplayers = len(start) - 3
                field = np.zeros((nplayers + 3, nplayers * 16), dtype='uint8')
                field[sparse_field] = 1
                skip=True
                for action_number in sparse_mask:
                    if is_capture(action_number, field, base, start)[0] is not None:
                        self.print(idx)
                        skip = False
                        break
            if not skip:
                self.print(idx)
                i += 1
            else:
                nskipped += 1
        print(f"Skipped {nskipped} samples")

    def __len__(self):
        if len(self.y) != len(self.X):
            print(f"GameData: Length of X, y arrays is not equal ({len(self.X)} vs {len(self.y)})")
            breakpoint()
        return len(self.X)

    def save_statistics(self, fname=None):
        print(f"==============================")
        print(f"Analyzing data set of size {len(self)}")
        player_names = []
        max_gl = 1000
        formats = {
                "int": ".0f",
                "ratio": ".3f",
                "arrM": None,
                "arrB": None,
        }
        entry_types = {
                # key: [type, label, normalization, showtransform, showformat]
            "ngames": ["int", "Games", None, "val", ".0f"],
            "own_move": ["arrM", "Player move", None, ["idxmean", "sum"], ".0f"],
            "opp_move": ["arrM", "Opponent's move", None, ["idxmean","sum"], ".0f"],
            "wins": ["int", "Wins", "ngames", "mean", ".3f"],
            "folds": ["int", "Folds", "own_move", "mean", ".3f"],
            "rounds": ["int", "Rounds", "ngames", "mean", ".0f"],
            "gamelengths": ["arrM", "Game length", None, "idxmean", ".0f"],
            "base": ["arrM", "Base occupancy", "total_move", "mean", ".2f"],
            "start": ["arrM", "Start occupancy", "total_move", "mean", ".2f"],
            "captures": ["arrB", "Captures", None, "sum", ".0f"],
            "potential_captures": ["arrB", "Potential captures", None, "sum", ".0f"],
            "captured": ["arrB", "Captured", None, "sum", ".0f"],
            "potential_captured": ["arrB", "Potentially captured", None, "sum", ".0f"],
            "captures_ratio": ["arrB", "Capture ratio", None, "mean", ".3f"],
            "captured_ratio": ["arrB", "Captured ratio", None, "mean", ".3f"],
            "activity": ["arrM", "Activity", None, "mean", ".3f"],
        }
        unique_keys = ["ngames", "rounds", "gamelengths"]
        ddict = {}
        def _instantiate(playername):
            ddict[playername] = {}
            for key, (value, _, _,_,_) in entry_types.items():
                if value in ["int", "ratio"]:
                    ddict[playername][key] = 0
                elif value == "dict":
                    ddict[playername][key] = {}
                elif value == "arrM":
                    ddict[playername][key] = np.zeros(max_gl)
                elif value == "arrB":
                    ddict[playername][key] = np.zeros(16*self.nplayers+1)
                else:
                    print(f"Unknown type {value} for key {key}")
                    breakpoint()
        _instantiate("totals")
        
        for i, (X, y) in enumerate(zip(self.X, self.y)):
            try:
                sparse_field, start, base, card, fold, roundn, playeridx, sparse_mask = X
                sparse_prob_idx, sparse_prob, y_val, move, l, playernames = y
                player = playernames[0]
                unique_playernames = list(set(playernames))
                oppnames = playernames[1:]
                for p in playernames:
                    if p not in player_names:
                        _instantiate(p)
                        player_names.append(p)
                pdict = ddict[player]
                pdict["own_move"][move] += 1
                best_action_number = sparse_prob_idx[np.argmax(sparse_prob)]
                field = np.zeros((self.nplayers + 3, self.nplayers * 16), dtype='uint8')
                field[sparse_field] = 1
                bestcap = False
                potcap = False
                captured = [False for i in range(self.nplayers)]
                for action_number in sparse_mask:
                    captured_player_idx, pos, possum, basesum,startsum = is_capture(action_number, field, base, start)
                    if captured_player_idx is not None:
                        captured_player = playernames[captured_player_idx]
                        if not bestcap and action_number == best_action_number:
                            bestcap = True
                            pdict["captures"][pos] += 1
                            ddict[captured_player]["captured"][pos] += 1
                        if not potcap:
                            pdict["potential_captures"][pos] += 1
                            potcap = True
                        if not captured[captured_player_idx]:
                            ddict[captured_player]["potential_captured"][pos] += 1
                            captured[captured_player_idx] = True
                if player == "Capture_best_action" and potcap and not bestcap:
                    mask = np.zeros(50).astype(bool)
                    mask[sparse_mask] = True
                    prob = np.zeros(50)
                    prob[sparse_prob_idx] = sparse_prob
                    nplayers = len(start) - 3
                    field = np.zeros((nplayers + 3, nplayers * 16), dtype='uint8')
                    field[sparse_field] = 1
                    print_action_space(mask, action_probabilities=prob, capture_info=[field, base, start])
                    breakpoint()

                if move == l:
                    pdict["wins"]+= 1
                    for p in unique_playernames:
                        ddict[p]["rounds"]+= roundn
                        ddict[p]["gamelengths"][l] += 1
                        ddict[p]["ngames"] += 1
                    ddict["totals"]["rounds"] += roundn
                    ddict["totals"]["gamelengths"][l] += 1
                    ddict["totals"]["ngames"] += 1
                    if self.nplayers == 2:
                        if y_val != 1:
                            print(f"Bad y_val {y_val} for last move of game")
                            breakpoint()
                    else:
                        if y_val != 0:
                            print(f"Bad y_val {y_val} i.c.w. player index {playeridx}")
                            breakpoint()
                if fold == 0:
                    for p in oppnames:
                        ddict[p]["folds"]+= 1 / (len(playernames) - 1)
                pdict["start"][move]+= np.sum(start[:4])
                pdict["base"][move]+= np.sum(base[:4])
                for j, name in enumerate(oppnames):
                    ddict[name]["start"][move]+= start[4 + j]
                    ddict[name]["base"][move]+= base[4 + j]
                    ddict[name]["opp_move"][move]+= 1
            except:
                traceback.print_exc()
                breakpoint()

        for name in player_names:
            pdict = ddict[name]
            pdict["activity"] = pdict["own_move"] / (pdict["opp_move"] + pdict["own_move"] + 1e-8)
            pdict["captures_ratio"] = pdict["captures"] / (pdict["potential_captures"] + 1e-8)
            pdict["captured_ratio"] = pdict["captured"] / (pdict["potential_captured"] + 1e-8)

        for name in player_names:
            for key in ddict[name].keys():
                if key in unique_keys:
                    continue
                ddict["totals"][key]+= ddict[name][key]

        maxmoveidx = np.nonzero(ddict["totals"]["own_move"])[0][-1]
        ext_player_names = ["totals", *player_names]
        
        metadata = ["mean", "idxmean", "sum", "idxsum", "std", "idxstd", "cumsum"]
        for name in ext_player_names:
            try:
                pdict = ddict[name]
                for mdata in metadata:
                    pdict[mdata] = {}
                for key in ddict[name].keys():
                    if key in metadata:
                        continue
                    # print(f"Doing key {key}")
                    _type, _, normtype,_,_ =  entry_types[key]
                    if _type in ["arrM", "arrB"]:
                        if _type == "arrM":
                            pdict[key] = pdict[key][:maxmoveidx]
                        kdict = pdict[key]
                        if key == "start":
                            pass
                            # breakpoint()

                        if normtype == "total_move":
                            kdict /= (pdict["opp_move"] + pdict["own_move"]) + 1e-8
                        elif normtype is not None:
                            if entry_types[normtype][0] in ["arrM", "arrB"]:
                                kdict /= pdict["sum"][normtype] + 1e-8
                            else:
                                kdict /= pdict[normtype] + 1e-8
                        pdict["sum"][key] = np.sum(kdict)
                        pdict["idxsum"][key] = np.sum(np.arange(len(kdict)) * kdict)
                        if key == "captured_ratio":
                            pdict["mean"]["captured_ratio"] = np.sum(pdict["captured"]) / np.sum(pdict["potential_captured"])
                        elif key == "captures_ratio":
                            pdict["mean"]["captures_ratio"] = np.sum(pdict["captures"]) / np.sum(pdict["potential_captures"])
                        elif key == "activity":
                            pdict["mean"]["activity"] = np.sum(pdict["own_move"]) / (np.sum(pdict["opp_move"]) + np.sum(pdict["own_move"]))
                        else:
                            pdict["mean"][key] = pdict["sum"][key] / len(np.nonzero(kdict)[0])
                        pdict["idxmean"][key] = pdict["idxsum"][key] / pdict["sum"][key]
                        pdict["std"][key] = np.sqrt(np.mean((kdict - pdict["mean"][key])**2) / len(np.nonzero(kdict)[0]))
                        pdict["idxstd"][key] = np.sqrt(np.mean(kdict/pdict["sum"][key] * (np.arange(len(kdict)) - pdict["mean"][key])**2) / pdict["sum"][key])
                        pdict["cumsum"][key] = np.cumsum(kdict)
                    elif _type in ["int", "ratio"]:
                        if normtype is not None:
                            if entry_types[normtype][0] in ["arrM", "arrB"]:
                                pdict["mean"][key] = pdict[key] / pdict["sum"][normtype]
                            else:
                                pdict["mean"][key] = pdict[key] / pdict[normtype]
            except Exception:
                traceback.print_exc()
                breakpoint()
        
        
        axinfo = [
                # title, xlabel, ylabel, twinylabel, ylim, twinylim
                ["Scalars", None, "Normalized to total", None, None, None],
                ["Last Move", "Move number", "Fraction of games", "Cumulative fraction of games", None, (0,1)],
                ["Start/Base Occupancy", "Move number", "Pawns", None, None, None],
                ["Captures", "Pawn position", "Fraction of moves", "Capture ratio", None, (0, 1)],
                ["Captured", "Pawn position", "Fraction of moves", "Captured ratio", None, (0, 1)],
                ]
        # axis, dictkey, twinax, linestyle
        datainfo = [
                [1, ["gamelengths", "divsum"], False, "solid"],
                [1, ["gamelengths", "cumsum"], True, "dotted"],
                [1, ["own_move", "divsum"], False, "dashed"],
                [2, ["start", None], False, "solid"],
                [2, ["base", None], False, "dotted"],
                [3, ["captures", None], False, "solid"],
                [3, ["captures_ratio", None], True, "dotted"],
                [4, ["captured", None], False, "solid"],
                [4, ["captured_ratio", None], True, "dotted"],
                ]

        nsubplot = len(axinfo)
        fig, axes = plt.subplots(nsubplot, figsize=(nsubplot*3, nsubplot*4))
        fig.suptitle(f"Statistics on GameData with {len(self)} samples", fontsize=18)
        twinaxes = [axes[i].twinx() for i in range(nsubplot)]
        color = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        player_colors = [next(color) for i in range(len(ext_player_names))]
        bar_keys = ["wins", "captures_ratio", "captured_ratio", "folds", "activity", "ngames", "own_move", "rounds"]
        bar_transforms = ["mean", "mean", "mean", "mean", "mean", None, "sum", "mean"]
        bar_norm_transforms = ["nplayers", None, None, None, None, None, "nplayers", None]
        bar_formats = [".3f", ".3f", ".3f", ".3f", ".3f", ".0f", ".0f", ".0f"]
        bar_labels = [entry_types[key][1] for key in bar_keys]
        bar_locs = np.arange(len(bar_keys))
        bar_width = 1 / (len(player_names) + 1)
        tick_locs = bar_locs + bar_width * (len(player_names)-1) / 2
        axes[0].set_xticks(tick_locs, bar_labels)
        for playeridx, player in enumerate(ext_player_names):
            try:
                pdict = ddict[player]
                cutoff_gl = max_gl
                for i, l in enumerate(pdict["cumsum"]["gamelengths"]):
                    if l > .95 * pdict["cumsum"]["gamelengths"][-1]:
                        cutoff_gl = i
                        break
                print(f"\nPlayer \033[1m{player}\033[0m:")
                for key, value in pdict.items():
                    if key in metadata:
                        continue
                # key: [type, label, normalization, showtransform, showformat]
                    _, valname, _, showlist, fmt = entry_types[key]
                    if showlist is None:
                        continue
                    if type(showlist) is not list:
                        showlist = [showlist]
                    totalstr = ""
                    for show in showlist:
                        descstr=""
                        if show == "val":
                            valstr = format(pdict[key], fmt)
                        else:
                            valstr = format(pdict[show][key], fmt)
                            if "mean" in show:
                                descstr = " (mean)"
                            elif "sum" in show:
                                descstr = " (sum)"
                        totalstr += f"{valstr:<7}{descstr:<5}, "
                    totalstr = totalstr[:-2]
                    print(f"\t{valname:<20}: {totalstr}")

                playeridx = ext_player_names.index(player)
                if player != "totals":
                    offset = bar_width * (playeridx - 1)
                    for i, (key, loc, tf, ntf, fmt) in enumerate(zip(bar_keys, bar_locs, bar_transforms, bar_norm_transforms, bar_formats)):
                        if tf == "sum":
                            val = pdict["sum"][key]
                            totalval = ddict["totals"]["sum"][key]
                        elif tf == "mean":
                            val = pdict["mean"][key]
                            totalval =ddict["totals"]["mean"][key]
                        else:
                            val = pdict[key]
                            totalval =ddict["totals"][key]
                        if ntf == "nplayers":
                            totalval /= len(player_names)
                        norm_val = val / totalval
                        valstr = format(totalval, fmt)
                        b = axes[0].bar(loc + offset, norm_val, bar_width, color=player_colors[playeridx], label=player)
                        if playeridx == len(ext_player_names) // 2:
                            axes[0].bar_label(b, [valstr], label_type='edge', padding=10)

                for axidx, (key, reduction), plottwin, linestyle in datainfo:
                    plotax = axes[axidx] if not plottwin else twinaxes[axidx]
                    if reduction is None:
                        var = pdict[key]
                    elif reduction == "divsum":
                        var = pdict[key] / pdict["sum"][key]
                    elif reduction == "cumsum":
                        var = pdict["cumsum"][key] / pdict["sum"][key]
                    else:
                        print(f"Unknown reduction type {reduction}")
                        breakpoint()
                    window_size = int(1/10 * len(var))
                    window = [1 / window_size] * window_size
                    smoothened_var = sl.convolve(var, window, mode='valid')
                    locs = np.arange(len(smoothened_var))
                    plot_idxs = np.nonzero(smoothened_var[:cutoff_gl])[0]
                    plotax.plot(locs[plot_idxs], smoothened_var[plot_idxs], linestyle=linestyle, label=player, color = player_colors[playeridx])

            except Exception:
                traceback.print_exc()
                breakpoint()
        for ax, twinax, (title, xlabel, ylabel, twinylabel, ylim, twinylim) in zip(axes,twinaxes, axinfo):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if twinylabel is not None:
                twinax.set_ylabel(twinylabel)
            else:
                twinax.set_yticks([])
            if ylim is not None:
                ax.set_ylim(ylim)
            if twinylim is not None:
                twinax.set_ylim(ylim)
        handles, labels = axes[1].get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        axes[0].legend(loc="upper right", handles=unique_handles, labels=unique_labels)
        fig.tight_layout()
        if fname is None:
            fname = f"figures/statistics/dataanalysis_{'-'.join(player_names)}_size_{len(self)}.png"
        fig.savefig(fname)
        print(f"==============================")


def main():
    data = GameData(nplayers=4)
    data.load_dir("data/4_players/Random_Capture_Smart_Eager_size_30037")
    # data.load_dir("/home/jer/Projects/machine_learning/keezenspel/data/2_players/TSPEnv_maxdepth-5_TSPEnv_maxdepth-5_size_300005")
    data.save_statistics()
    # data.show_random(capture_only=True)

if __name__ == '__main__':
    main()

