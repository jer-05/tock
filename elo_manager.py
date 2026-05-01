from dataclasses import dataclass
import math
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import random
import numpy as np
import itertools
import pickle

BASE_ELO_PLAYERS = {"Smart": 250, "Random": 100, "TSPDet": 350, "NN": 200, "NNMCTS": 300, "TSPEnv": 500, "Challenger": 300}
COLOR_MAP = {"Challenger": "blue", "TSPDet": "red", "Random": "grey", "Smart": "orange", "NNMCTS": "deepskyblue", "NN": "green", "TSPEnv": "purple"}
class EloManager:
    def __init__(self, nplayers, c_elo=400, K=15, base_elo=250, verbose=False, challenger_id=None):
        print(f"Initialized Elo Manager for {nplayers} players")
        self.nplayers = nplayers
        if self.nplayers != 2:
            raise NotImplementedError
        self.c_elo = c_elo
        self.update_times = []
        self.challenger_hash = ("Challenger", "current_weights")
        self.challenger_id = challenger_id
        self.verbose = verbose
        self.K = K
        self.minK = float(K) / 4
        self.maxK = 4 * K
        self.max_ci = c_elo / 2
        self.min_ci = c_elo / 60
        self.base_elo=100
        self.time = 0
        self.min_std_window = 90
        self.std_window = self.min_std_window
        self.match_results = {}
        self.evolve_times = []
        self.c_ci = 2
        self.alpha = .05
        self.time_his = []
        self.prov_elo_updates = self.std_window // 2
        self.pboost = 4
        self.alpha_window = .2
        self.min_play_weight = 1 / (self.max_ci)
        self.max_play_weight = 1 / (self.min_ci)
        self.players = {}
        self.aliases = {}
        self.add_player(*self.challenger_hash)
        print(self)

    def duplicate_player(self, name, config, newname, newconfig):
        hashkey = self.get_hash(name, config, alias=True)
        new_hashkey = self.get_hash(newname, newconfig)
        self.players[new_hashkey] = copy.deepcopy(self.players[hashkey])
        self.players[new_hashkey]["config"] = copy.deepcopy(newconfig)
        self.players[new_hashkey]["name"] = newname
        self.players[new_hashkey]["duplication_time"] = self.time
        if self.verbose:
            print(f"EloManager: duplicated players:\n   {hashkey}\nD->{new_hashkey}") 

    def get_weights(self, name, config):
        phash = self.get_hash(name, config, alias=True)
        if phash is None:
            if self.verbose:
                print(f"Player {name}, {config} does not exist yet; returning maximum weight of {self.max_play_weight:.3f}")
            return self.max_play_weight
        ngames = self.players[phash]["ngames"]
        if ngames < self.prov_elo_updates:
            if self.verbose:
                print(f"Player {name}, {config} only played {ngames} < {self.prov_elo_updates} games; returning minimum weight of {self.min_play_weight:.3f}")
            return self.min_play_weight
        ci = self.players[phash]["ci"]
        weight = 1 / (ci[1] - ci[0])
        weight = np.clip(weight, self.min_play_weight, self.max_play_weight)
        return weight


    def split_off_player(self, name, config, newname, newconfig):
        self.duplicate_player(name, config, newname, newconfig)
        dup_dict = self.players[self.get_hash(newname, newconfig, alias=True)]
        self.add_player(name, config, elo=self[name, config][0], overwrite=True)
        self.players[self.get_hash(name, config, alias=True)]["split_his"] = {"elo_his": dup_dict["elo_his"][:], "ngames_his": dup_dict["ngames_his"][:], "ci_his": copy.deepcopy(dup_dict["ci_his"]), "time_his": dup_dict["time_his"][:], "creation_time": dup_dict["creation_time"], "elo": dup_dict["elo"]}
        if "split_his" in dup_dict:
            self.players[self.get_hash(name, config, alias=True)]["split_his"]["split_his"] = copy.deepcopy(dup_dict["split_his"])
        if self.verbose:
            print(f"EloManager: split off player {self.get_hash(name, config, alias=True)} to (new) player {self.get_hash(newname, newconfig, alias=True)}")

    def plusevolve(self):
        self.evolve_times.append(self.time)

    def add_alias(self, name, config, aliasname, aliasconfig):
        """
        First put the existing player's name. Then the new alias.
        """
        hashkey = self.get_hash(name, config)
        if hashkey not in self.aliases.keys():
            if hashkey not in self.players.keys():
                print(f"EloManager: trying to add alias for player {hashkey}, but player does not exist!")
                breakpoint()
        alias_hashkey = self.get_hash(aliasname, aliasconfig)
        self.aliases[alias_hashkey] = hashkey
        if self.verbose:
            print(f"EloManager: aliased players:\n   {hashkey}\nA->{alias_hashkey}")
        
    def add_player(self, name, config, *, elo=None, reference_key=None, overwrite=False, softfail_overwrite=False):
        if elo is None and reference_key is None:
            if name in BASE_ELO_PLAYERS:
                elo = BASE_ELO_PLAYERS[name]
            else:
                elo = self.base_elo
        elif reference_key is not None:
            name, config = reference_key
            if config is None:
                used_name = True
                phash = None
                for key, value in self.players.items():
                    if value["name"] == name:
                        phash = key
                        break
            else:
                used_name = False
                phash = self.get_hash(*reference_key, alias=True)
            if phash is None:
                print("EloManager.add_player: Bad reference key?")
                print(f"Non-existing key is {reference_key if not used_name else name}")
                breakpoint()
            elo = self.players[phash]["elo"]
            if self.verbose:
                print(f"EloManager.add_player: using reference hash {phash} and initialised elo to {elo}") 
        ci = np.array([elo - .5 * self.max_ci, elo + .5 * self.max_ci])
        hashkey = self.get_hash(name, config)
        if (hashkey in self.players.keys() or hashkey in self.aliases.keys()):
            if overwrite:
                if self.verbose:
                    print(f"EloManager.add_player: Overwriting player !!!")
                while hashkey in self.aliases.keys():
                    hashkey = self.aliases[hashkey]
                name = self.players[hashkey]["name"]
                config = self.players[hashkey]["config"]
            elif softfail_overwrite:
                print(f"EloManager.add_player: add_player failed because player with hash {hashkey} exists")
                return
            else:
                print(f"EloManager.add_player: trying to add existing player")
                breakpoint()
        self.players[hashkey] = {"elo": elo, "ci": ci, "ngames": 0, "name": name, "config": config, "ngames_his": [], "elo_his": [], "ci_his": [], "time_his": [], "creation_time": self.time}
        if self.verbose:
            print(f"Elo Manager: Added new player with hashkey {hashkey}")

    def get_hash(self, *args, alias=False):
        def hashable(s):
            # print(f"s is {s}")
            if isinstance(s, dict):
                return tuple((key, hashable(value)) for key, value in sorted(s.items()))
            elif isinstance(s, list) or isinstance(s, tuple):
                return tuple([hashable(item) for item in s])
            return s
        if self.challenger_id is not None:
            challenger_config_key = list(self.challenger_id.keys())[0]
            if challenger_config_key in args[1]:
                ch_config_val = list(self.challenger_id.values())[0]
                if ch_config_val in args[1]["fname"]:
                    return self.challenger_hash
        playerhash = hashable(args)
        if alias:
            while playerhash not in self.players:
                try:
                    playerhash = self.aliases[playerhash]
                except KeyError:
                    return None
        # print(f"Returning hash {playerhash}")
        return playerhash


    def __getitem__(self, key):
        name, config = key
        hashkey = self.get_hash(name, config)
        if hashkey not in self.players:
            alias_hashkey = self.aliases[hashkey]
            return self[alias_hashkey]
        return self.players[hashkey]['elo'], self.players[hashkey]['ci']

    def add_match_result(self, names, configs, wins):
        if self.verbose:
            print("EloManager: adding match results")
        keys = [self.get_hash(name, config) for name, config in zip(names,configs)]
        for i, hashkey in enumerate(keys):
            if not (hashkey in self.players or hashkey in self.aliases.keys()):
                self.add_player(names[i], configs[i])
            if not hashkey in self.players:
                while not hashkey in self.players:
                    hashkey = self.aliases[hashkey]
                keys[i] = hashkey
        match_id = self.get_hash(names, configs)
        if match_id not in self.match_results.keys():
            self.match_results[match_id] = {"names": names, "configs": configs, "wins": wins}
            if self.verbose:
                print(f"Added match result with wins {wins} for names {names}, configs {configs}")
        else:
            if self.verbose:
                print(f"Updating match result with wins {wins} for names {names}, configs {configs}")
                print(f"Existing match result already has {self.match_results[match_id]['wins']} wins")
            self.match_results[match_id]["wins"] += wins
    def update_elo(self):
        if self.verbose:
            print(f"==========================")
        if self.match_results == {}:
            if self.verbose:
                print(f"EloManager: no matches to process")
                print(f"==========================")
            return
        update_time = self.time
        self.update_times.append(update_time)
        match_results = [_ for _ in self.match_results.values()]
        names = [mr["names"] for mr in match_results]
        configs = [mr["configs"] for mr in match_results]
        if self.verbose:
            print(f"EloManager: Updating Elos for {len(match_results)} matches at time {update_time}")
        nresults = len(match_results)
        ngames = np.array([sum(res["wins"]) for res in match_results], dtype=float)
        pcts = np.array([res["wins"][0] / ngames[i] for i, res in enumerate(match_results)])
        sds = np.array([math.sqrt(pct*(1-pct) / ng) for pct, ng in zip(pcts, ngames)])
        total_games = sum(ngames)
        weights = ngames / total_games
        idxs = random.choices(range(nresults), weights=list(weights), k=10*int(total_games))
        def pp_arr(arr, fmt, sep, leader):
            print(f"{leader}: [{sep.join([format(x, fmt) for x in arr])}]")
        
        updated_games = [0 for i in range(nresults)]

        for idx in idxs:
            if sum(updated_games) == total_games:
                break
            if updated_games[idx] >= ngames[idx]:
                continue
            mr = match_results[idx]
            keys = [self.get_hash(name, config, alias=True) for name, config in zip(names[idx], configs[idx])]
            self.one_game_elo_update(keys, [pcts[idx], 1-pcts[idx]])
            updated_games[idx] += 1
        self.match_results = {}
        if self.verbose:
            leaders = ["names", "  n ", "  % ", "  sd", " games", "  w "]
            arrs = [names, ngames, 100*pcts, 100*sds, updated_games, weights]
            fmts = ["", ".0f", ".1f", ".2f", "", ".3f"]
            print(f"Match results to be processed are:")
            for arr, fmt, leader in zip(arrs, fmts, leaders):
                pp_arr(arr, fmt, ", ", leader)
        assert sum(updated_games) == total_games
        if self.verbose:
            print(f"EloManager: processed {int(total_games)} games. Elos changed:")
            for key in self.players.keys():
                print(self.get_player_str(key, old_time=update_time), end='')
        delta_update_time = self.time - update_time
        new_window = max(self.min_std_window, delta_update_time // 2)
        self.std_window = round(self.alpha_window * new_window + (1 - self.alpha_window) * self.std_window)
        self.prov_elo_updates = self.std_window // 2
        if self.verbose:
            print(f"Set window for calculation of CI to {self.std_window} and number of provisional games to {self.prov_elo_updates}")
            print(f"==========================")
        else:
            print(f"EloManager: Processed match results")
            
    def check_ci(self, _min, _max, ci, mean):
        if not (mean <= ci[1] and mean >= ci[0] and ci[0] >= _min and ci[1] <= _max):
            if not (np.isclose(mean, ci[1]) or np.isclose(mean, ci[0])):
                print("EloManager.update_elo: Bad confidence interval?")
                print(f"Value is {mean:.3f} with CI [{ci[0]:.3f}, {ci[1]:.3f}], (min: {_min}, max: {_max})") 
                breakpoint()

    def one_game_elo_update(self, keys, pwin):
        assert len(keys) == 2, "EloManager.update_elo:  incorrect number of players"
        pdicts = [self.players[keys[i]] for i in range(2)]
        for i in range(2):
            pdict = pdicts[i]
            ngames = pdict["ngames"]
            prov_fact_opp = min(1, pdicts[(i + 1) % 2]["ngames"] / self.prov_elo_updates)
            prov_fact_self = min(1, ngames / self.prov_elo_updates)
            prov_fact = 1 - prov_fact_self + prov_fact_opp
            if ngames < self.prov_elo_updates:
                ci_width = self.max_ci
                prov_fact_opp = 2
            else:
                ci = pdict["ci"]
                ci_width = ci[1] - ci[0]
            
            K = self.K * 1/ np.sqrt(1 + ngames) *  ci_width / self.max_ci * prov_fact
            K = np.clip(K, self.minK, self.maxK)
            current_elo = pdict["elo"]
            expected_p = 1 / (1 + 10 ** ((pdicts[(i + 1) % 2]["elo"] - current_elo) / self.c_elo))
            delta_p = pwin[i] - expected_p
            delta = K * delta_p
            new_elo = current_elo + delta
            new_ngames = pdict["ngames"] + 1
            elo_his = pdict["elo_his"]
            old_ci = pdict["ci"]
            if elo_his:
                m = self.c_ci * np.std(elo_his[max(0, len(elo_his) - self.std_window):])
                ngames = pdict["ngames"]
                new_elo_CI = np.array([new_elo - m, new_elo + m])
                updated_CI = self.alpha * new_elo_CI + (1-self.alpha) * old_ci
                updated_CI[0] = np.clip(updated_CI[0], new_elo - .5 * self.max_ci, new_elo - .5 * self.min_ci)
                updated_CI[1] = np.clip(updated_CI[1], new_elo + .5*self.min_ci, new_elo + .5*self.max_ci)
            else:
                updated_CI = old_ci
            self.check_ci(-np.inf, np.inf, updated_CI, new_elo)
            pdict["ci"] = updated_CI
            pdict["ci_his"].append(pdict["ci"])
            pdict["elo_his"].append(current_elo)
            pdict["ngames_his"].append(pdict["ngames"])
            pdict["elo"] = new_elo
            pdict["ngames"] = new_ngames
            pdict["time_his"].append(self.time)
        self.time_his.append(self.time)
        self.time += 1


    def plot_evolutions_and_updates(self, ax, legend_plot=False):
        for i, time in enumerate(self.evolve_times):
            ax.axvline(x=time, color="black", linestyle=":", label="Evolution")
            ax.text(x=time, y=1.03 * ax.get_ylim()[1], s=f"{i}", va='top', ha='center')
            ticks = list(ax.get_xticks())
            update_ticks = self.update_times
            update_labels = ["^"] * len(update_ticks)
            tick_labels = [int(tick) for tick in ticks]
            ax.set_xticks(ticks + update_ticks, tick_labels + update_labels)
        lines=[]
        lines.append(Line2D([], [], color="none", markerfacecolor="black", marker="x", label="Creation"))
        lines.append(Line2D([], [], color="none", markerfacecolor="black", marker="v", label="Split"))
        update_handle = Line2D([], [], 
                       marker=r'$\wedge$', 
                       color='black', 
                       linestyle='None', 
                       label='Start Elo Update') 
        lines.append(update_handle)
        lines.append(ax.axvline(x=0, color="black", linestyle=":", label="Evolution"))
        if legend_plot:
            for line in lines:
                ax.add_line(line)
        return lines

    def plot_all_elo_progression(self, *, filemanager=None, fname=None):
        other_c = itertools.cycle(["brown", "gold", "chocolate"])

        fig, ax = plt.subplots(figsize=(15, 9))
        fig.suptitle("Elo progression for all players over time")
        twin_ax = ax.twinx()
        twin_ax.set_ylabel("Games played")
        axes = [ax, twin_ax]
        ax.set_xlabel("Total games played")
        ax.set_ylabel("Elo")
        color_legend = []
        added_pairs = {}
        ch_name = self.players[self.challenger_hash]["name"]
        ch_config = self.players[self.challenger_hash]["config"]
        for i, pdict in enumerate(self.players.values()):
            name = pdict["name"]
            if name in COLOR_MAP:
                color = COLOR_MAP[name]
            else:
                color = next(other_c)
            config = pdict["config"]
            if (name, config) == (ch_name, ch_config):
                continue
            else:
                self.plot_elo_progression(name, config, color=color, axes=axes)
            
            if (name, color) not in added_pairs:
                color_legend.append(Patch(facecolor=color, edgecolor=color, label=name))
            added_pairs[(name, color)] = True
        self.plot_elo_progression(ch_name, ch_config, color=COLOR_MAP["Challenger"], axes=axes, plot_split=True)
        color_legend.append(Patch(facecolor=COLOR_MAP["Challenger"], edgecolor=COLOR_MAP["Challenger"], label="Challenger"))
        color_legend = ax.legend(title="Players", handles=color_legend, loc = "center left", bbox_to_anchor = (1.07, .5))
        ax.add_artist(color_legend)
        lines = []
        lines.append(Line2D([0], [0], linestyle="solid", color="black", label="Elo"))
        lines.append(Line2D([0], [0], linestyle="dashed", color="black", label="CI"))
        lines.append(Line2D([0], [0], color="black", linestyle="dotted", label="Games"))
        lines += self.plot_evolutions_and_updates(ax)
        labels = [line.get_label() for line in lines]
        leg1 = ax.legend(lines, labels, loc='center left', bbox_to_anchor = (1.07, .9))
        ax.add_artist(leg1)

        fig.tight_layout()
        fig.subplots_adjust(right=.8)
        if filemanager is None:
            if fname:
                fig.savefig(fname)
            else:
                fig.savefig(f"figures/full_elo_progression.png")
        else:
            idd= f"full_elo_progression"
            filemanager.save(fig, "elofig", idd=idd)
        print(f"Plotted full Elo progression")

    def plot_elo_progression(self, name, config, *, color=None, axes=None, filemanager=None, plot_split=False, fname=None):
        if color is None:
            colors = {"ci": "green", "elo": "blue", "ngames": "orange"}
        else:
            colors = {"ci": color, "elo": color, "ngames": color}
        player_hash = self.get_hash(name, config, alias=True)
        pdict = self.players[player_hash]
        dicts = []
        curr_dict = pdict
        if plot_split:
            while "split_his" in curr_dict:
                dicts.append(curr_dict)
                curr_dict = curr_dict["split_his"]
            dicts.append(curr_dict)
        else:
            dicts = [pdict]
        if axes is None:
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.suptitle(f"Elo progression of player {name}")
            twin_ax = ax.twinx()
        else:
            ax = axes[0]
            twin_ax = axes[1]
        lines = []
        for dictidx, pdict in enumerate(dicts):
            time = pdict["time_his"]
            start_idx = 0
            for i, t in enumerate(time):
                if t >= pdict["creation_time"]:
                    start_idx = i
                break
            if t == time[-1]:
                    start_idx = i
                    break
            time = time[start_idx:]
            elo_his = pdict["elo_his"][start_idx:]

            elo_ci_his = pdict["ci_his"][start_idx:]
            ngames_his = pdict["ngames_his"][start_idx:]
            ax.plot(time, elo_his, linestyle="solid", color=colors["elo"], label="Elo")[0]
            ax.plot(time, elo_ci_his, linestyle="dashed", color=colors["ci"], label="Elo CI")[0]
            twin_ax.plot(time, ngames_his, color=colors["ngames"], linestyle="dotted", label="Games Played")[0]
            xy = (pdict["creation_time"], elo_his[0] if elo_his != [] else pdict["elo"])
            if not "split_his" in pdict:
                ax.scatter(*xy, marker="x", color="black", label="Creation")
            else:
                ax.scatter(*xy, marker="v", color="black", label="Split")
        if axes is None:
            self.plot_evolutions_and_updates(ax, legend_plot=True)
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = twin_ax.get_legend_handles_labels()
            by_label = dict(zip(labels + labels2, handles + handles2))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            fig.tight_layout()
            if filemanager is None:
                if fname is None:
                    fig.savefig(f"figures/elo_progression_{name}.png")
                else:
                    fig.savefig(f"{fname}")
            else:
                idd= f"elo_progression_{name}"
                filemanager.save(fig, "elofig", idd=idd)
            plt.close('all')
            print(f"Plotted Elo progression of player {name}")

    def get_player_str(self, key, old_time=None):
        pdict = self.players[key]
        config = pdict["config"]
        name = pdict["name"]
        time_dict = pdict["time_his"]
        cfg_str = f"      config: {config}"
        creation_str = f"      created: time {pdict['creation_time']}"
        olddata = False
        if old_time is not None:
            i = -1
            for i in range(len(time_dict)):
                if old_time <= time_dict[i]:
                    idx = i
                    olddata = True
                    break
        if olddata:
            old_elo, old_ngames = pdict["elo_his"][i], pdict["ngames_his"][i]
            noplay = False
            old_ci = pdict["ci_his"][i]
            elo_str = f"      elo   : {old_elo:.1f} [{old_ci[0]:.1f}, {old_ci[1]:.1f}] -> {pdict['elo']:.1f} [{pdict['ci'][0]:.1f}, {pdict['ci'][1]:.1f}]"
            ngames_str =  f"      ngames: {round(old_ngames)} -> {round(pdict['ngames'])}"
        elif old_time is not None and not olddata:
            return ""
        else:
            ngames = pdict["ngames"]
            elo_str = f"      elo   : {pdict["elo"]:.1f} [{pdict["ci"][0]:.1f}, {pdict["ci"][1]:.1f}]"
            ngames_str =  f"      ngames: {round(pdict['ngames'])}"
        if pdict['ngames'] < self.prov_elo_updates:
            provisional_str = " (provisional)"
        else:
            provisional_str = ""

        s = f"    {name}:\n{cfg_str}\n{elo_str}{provisional_str}\n{ngames_str}\n{creation_str}\n"
        return s

    def get_aliases_str(self):
        s = "\nEloManager Aliases:\n"
        if self.aliases == {}:
            s += "  No defined aliases\n"
            return s
        visited_hashes = []
        for i, alias in enumerate(reversed(self.aliases.keys())):
            if alias in visited_hashes:
                continue
            alias_str = f"  {alias}"
            curr_hash = alias
            d = 0
            while curr_hash not in self.players:
                d += 1
                visited_hashes.append(curr_hash)
                curr_hash = self.aliases[curr_hash]
                alias_str += f" -> {curr_hash}"
            s += alias_str + "\n"
        return s


    def __repr__(self):
        s = f"==================================\n"
        s += f"EloManager Constants:\n"
        s += f"  Elo Constant:                     : {self.c_elo}\n"
        s += f"  Elo Update Factor (K)             : {self.K}\n"
        s += f"  Base Elo                          : {self.base_elo}\n"
        s += f"  Minimum CI calculation window     : {self.min_std_window}\n"
        s += f"  Alpha window                      : {self.alpha_window}\n"
        s += f"  Alpha CI interval                 : {self.alpha}\n"
        s += f"  Minimum CI interval               : [-{self.min_ci:.1f}, +{self.min_ci:.1f}]\n"
        s += f"  Maximum CI interval               : [-{self.max_ci:.0f}, +{self.max_ci:.0f}]\n"
        s += f"\nEloManager Variables:\n"
        s += f"  Total games played                : {self.time}\n"
        s += f"  Current provisional game threshold: {self.prov_elo_updates}\n"
        s += f"  Current CI calculation window     : {self.std_window}\n"
        s += f"\nEloManager Players:\n"
        i=0
        for key, value in self.players.items():
            i += 1
            s += self.get_player_str(key)
        if i == 0:
            s += f"   no registered players\n"
        s += self.get_aliases_str()
        s += f"=================================="
        return s


import numpy as np
if __name__ == '__main__':
    em = EloManager(nplayers=2, verbose=True)
    em.add_player("jeroen", {})
    em.add_player("bart", {})
    em.add_alias("bart", {}, "newbart", {})
    for i in range(10):
        em.add_match_result(["bart", "NNMCTS"], [{}, {}], [100, 100])
        em.add_match_result(["jeroen", "bartMCTS"], [{}, {}], [random.randint(0, 100), random.randint(0, 100)])
        em.add_match_result(["jeroen", "bart"], [{}, {}], [random.randint(0, 100), random.randint(0, 100)])
        em.add_match_result(["jeroen", "bart"], [{}, {}], [100, 200])
        em.add_match_result(["jeroen", "TSPEnv"], [{}, {}], [140, 200])
        em.add_match_result(["jeroen", "Random"], [{}, {}], [130, 200])
        em.add_match_result(["TSPEnv", "suus"], [{}, {}], [130, 1000])
        em.add_match_result(["suus", "jeroen"], [{}, {}], [130, 1000])
        em.add_alias("bart", {}, f"newbart{i}", {})
        em.plusevolve()
        em.update_elo()
        em.plot_all_elo_progression()
        em.plot_elo_progression("bart", {}, plot_split=True)
        em.split_off_player("newbart", {}, "bart_split", {})
        print(em)
    em.plot_all_elo_progression(challenger=("bart", {}))
    print(em)
    with open("emdump", "wb") as f:
        pickle.dump(em, f)
    with open("emdump", "rb") as f:
        dupem = pickle.load(f)
    print(dupem)





        


