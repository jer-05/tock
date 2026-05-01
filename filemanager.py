import os
import torch
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

SESSION_DIR = "training_sessions"
DIRMAP = {
        "cpt": "figures/competition",
        "train": "figures/training",
        "wcurr": "weights/current",
        "wbest": "weights/best",
        "data": "data/datasets",
        "elofig": "figures/elo",
        "hpp": "hyperparameters",
        "elo": "elo",
        "elodump": "elo/dump",
        "hppdump": "hyperparameters/dump",
        "elotxt": "elo/txt",
        "hpptxt": "hyperparameters/txt",
        "dataage": "data/age",
}
DIRTYPE = ["data"]
FILEMAP = {
        "cycle": "cycle.txt",
        "evo": "evo.txt",
        "round": "round.txt",
}

class TrainingFileManager:
    def __init__(self, restore_idx=None, verbose=False):
        self.verbose=verbose
        i=0
        if restore_idx is None:
            while True:
                self.dir = f"{SESSION_DIR}/session{i}"
                if not os.path.isdir(self.dir):
                    os.makedirs(self.dir)
                    break
                i += 1
            self.evo_ctr = 0
            self.cycle_ctr = 0
            self.round_ctr = 0
            self.write_ecr()
        else:
            if restore_idx == -1:
                session_names = os.listdir(SESSION_DIR)
                ordered = sorted(session_names, key = lambda x: os.path.getmtime(os.path.join(SESSION_DIR, x)))
                self.dir = f"{SESSION_DIR}/{ordered[-1]}"
            else:
                self.dir = f"{SESSION_DIR}/session{restore_idx}"
            if not os.path.isdir(self.dir):
                print(f"Cannot restore session at index {restore_idx}; directory {self.dir} does not exist")
                breakpoint()
            else:
                print(f"Restored file manager from session {self.dir}")
            self.read_ecr()
        for dirname in DIRMAP.values():
            full_path = f"{self.dir}/{dirname}"
            if not os.path.isdir(full_path):
                os.makedirs(full_path)
        self.print()

    def get_cycle_file(self):
        return f"{self.dir}/{FILEMAP['cycle']}"

    def write_ecr(self):
        with open(f"{self.dir}/{FILEMAP['evo']}", "w") as f:
            f.write(str(self.evo_ctr))
        with open(f"{self.dir}/{FILEMAP['cycle']}", "w") as f:
            f.write(str(self.cycle_ctr))
        with open(f"{self.dir}/{FILEMAP['round']}", "w") as f:
            f.write(str(self.round_ctr))


    def read_ecr(self):
        with open(f"{self.dir}/{FILEMAP['evo']}", "r") as f:
            evo = int(f.read())
        with open(f"{self.dir}/{FILEMAP['cycle']}", "r") as f:
            cycle = int(f.read())
        with open(f"{self.dir}/{FILEMAP['round']}", "r") as f:
            _round = int(f.read())
        self.evo_ctr = evo
        self.cycle_ctr = cycle
        self.round_ctr = _round

    def save(self, data, dirid, *, _fname=None, idd=None):
        fname = self.getname(dirid, _fname, idd)
        if dirid in ["fig", "train", "cpt", "elofig"]:
            data.savefig(fname)
        elif dirid in ["wcurr", "wbest", "weight", "dataage"]:
            torch.save(data, fname)
        elif dirid == "data":
            data.save(fname)
        elif dirid in ["hpp"]:
            fname_dump = self.getname(dirid, _fname, idd, subdir="dump")
            fname_txt = self.getname(dirid, _fname, idd, subdir="txt")
            with open(fname_txt, "w") as f:
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            with open(fname_dump, "wb") as f:
                pickle.dump(data, f)
        elif dirid in ["elo"]:
            fname_dump = self.getname(dirid, _fname, idd, subdir="dump")
            fname_txt = self.getname(dirid, _fname, idd, subdir="txt")
            with open(fname_dump, "wb") as f:
                pickle.dump(data, f)
            with open(fname_txt, "w") as f:
                print(data, file=f)
        else:
            print(f"TrainingFileManager: unknown dirid {dirid}")
            breakpoint()
        if self.verbose:
            print(f"Saved data to {fname}")

    def getname(self, dirid, fname=None, idd=None, subdir=None):
        dirname = self.getdirname(dirid)
        if subdir is not None:
            dirname = dirname + f"/{subdir}"
        if fname is None:
            idd_str = "" if idd is None else f"_{idd}"
            fname = f"evo_{self.evo_ctr}_cycle_{self.cycle_ctr}_round_{self.round_ctr}{idd_str}"
        fname = f"{dirname}/{fname}"
        if subdir == "txt":
            fname = f"{fname}.txt"
        elif subdir == "dump":
            fname = f"{fname}.pkl"
        elif dirid in ["fig", "train", "cpt"]:
            fname = f"{fname}.png"
        elif dirid in ["wcurr", "wbest", "weight", "dataage"]:
            fname = f"{fname}.pth"
        i=0
        extended_fname = fname
        if "." in fname:
            extension_idx = fname.index(".")
        else:
            extension_idx = len(fname)
        while os.path.isfile(extended_fname):
            extended_fname = fname[:extension_idx] + f"_idx_{i}" + fname[extension_idx:]
            i+=1
        return extended_fname

    def getdirname(self, dirid):
        if dirid not in DIRMAP:
            print(f"TrainingFileManager.getdirname: unknown dirid '{dirid}'")
            breakpoint()
        dirname = f"{self.dir}/{DIRMAP[dirid]}"
        return dirname

    def getall(self, dirid):
        dirname = self.getdirname(dirid)
        fnames = os.listdir(dirname)
        sorted_fnames = sorted(fnames, key = lambda x: os.path.getmtime(os.path.join(dirname, x)))
        return [f"{dirname}/{sorted_fnames[i]}" for i in range(len(sorted_fnames))]

    def getlatest(self, dirid, soft_fail=False):
        dirname = self.getdirname(dirid)
        dir_entry_names = os.listdir(dirname)
        if dirid not in DIRTYPE:
            fnames = [entry for entry in dir_entry_names if os.path.isfile(f"{dirname}/{entry}")]
        else:
            fnames = dir_entry_names

        if not len(fnames):
            if soft_fail:
                return None
            else:
                print(f"TrainingFileManager: dir with dirid {dirid} has no files and soft_fail not set")
                breakpoint()
        sorted_fnames = sorted(fnames, key = lambda x: os.path.getmtime(os.path.join(dirname, x)))
        return f"{dirname}/{sorted_fnames[-1]}"
    
    def pluscycle(self):
        self.cycle_ctr += 1
        self.round_ctr = 0
        self.write_ecr()
        if self.verbose:
            print(f"TrainingFileManager: updated cycle to {self.cycle_ctr}")

    def plusround(self):
        self.round_ctr += 1
        self.write_ecr()
        if self.verbose:
            print(f"TrainingFileManager: updated round to {self.round_ctr}")

    def plusevolve(self):
        self.evo_ctr += 1
        self.cycle_ctr = 0
        self.round_ctr = 0
        self.write_ecr()
        if self.verbose:
            print(f"TrainingFileManager: updated evo cycle to {self.evo_ctr}")
            print(f"TrainingFileManager: reset cycle to {self.cycle_ctr}")
            print(f"TrainingFileManager: reset round to {self.round_ctr}")

    def print(self):
        print("--------------------")
        print(f"TrainingFileManager:")
        print(f"  Evolution number: {self.evo_ctr}")
        print(f"  Cycle number    : {self.cycle_ctr}")
        print(f"  Round number    : {self.round_ctr}")
        print(f"  File structure  :", flush=True)
        os.system(f"tree {self.dir}")
        print(f"  Latest values:")
        for dirid in DIRMAP.keys():
            latest = self.getlatest(dirid, soft_fail=True)
            print(f"    {dirid}: {latest}")
        print(f"-------------------")

def main():
    tfm = TrainingFileManager()
    t = torch.Tensor([3,4,5])
    fig, ax = plt.subplots()
    ax.plot([2,3], [4,5])
    hpp = {"dummy": True}
    for evo in range(3):
        for cycle in range(3):
            tfm.save(t, "wcurr")
            tfm.save(t, "wbest")
            tfm.save(fig, "cpt")
            tfm.save(fig, "train")
            tfm.save(fig, "fig")
            tfm.save(hpp, "hpp")
            tfm.save(hpp, "hpp")
            tfm.save(hpp, "hpp")
    tfm.print()

if __name__ == "__main__":
    main()


