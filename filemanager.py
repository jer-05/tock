import os
import torch
import matplotlib.pyplot as plt

SESSION_DIR = "training_sessions"
FIG_DIR = "figures"
CPT_FIG_DIR = f"{FIG_DIR}/competition"
TRAIN_FIG_DIR = f"{FIG_DIR}/training"
WEIGHTS_DIR = "weights"
W_CURR_DIR = f"{WEIGHTS_DIR}/current"
W_BEST_DIR = f"{WEIGHTS_DIR}/best"
DATA_DIR = "data"


class TrainingFileManager:
    def __init__(self):
        i=0
        while True:
            sdir = f"{SESSION_DIR}/session{i}"
            if not os.path.isdir(sdir):
                os.system(f"mkdir {sdir}")
                break
            i += 1
        self.dir = sdir
        figdir = f"{sdir}/{FIG_DIR}"
        cptfigdir = f"{sdir}/{CPT_FIG_DIR}"
        trainfigdir = f"{sdir}/{TRAIN_FIG_DIR}"
        wdir = f"{sdir}/{WEIGHTS_DIR}"
        wcurrdir = f"{sdir}/{W_CURR_DIR}"
        wbestdir = f"{sdir}/{W_BEST_DIR}"
        datadir = f"{sdir}/{DATA_DIR}"
        self.dirs = {
                "fig": figdir,
                "weight": wdir,
                "cpt": cptfigdir,
                "train": trainfigdir,
                "wcurr": wcurrdir,
                "wbest": wbestdir,
                "data": datadir,
        }
        self.latest = {
                "fig": "",
                "weight": "",
                "cpt": "",
                "train": "",
                "wcurr": "",
                "wbest": "",
                "data": "",
        }
        for dirname in self.dirs.values():
            os.system(f"mkdir {dirname}")
        self.evo_ctr = 0
        self.cycle_ctr = 0
        print("Successfully initialized file manager.")
        self.print()

    def save(self, data, dirid, *, fname=None, idd=None, verbose=False):
        fname = self.getname(dirid, fname, idd)
        if os.path.isfile(fname):
            print(f"Warning in 'save': File {fname} Exists, overwriting now!")
        if dirid in ["fig", "train", "cpt"]:
            data.savefig(fname)
        elif dirid in ["wcurr", "wbest", "weight", "data"]:
            torch.save(data, fname)
        if verbose:
            print(f"Saved data to {fname}")

    def getname(self, dirid, fname=None, idd=None):
        dirname = self.getdirname(dirid)
        if dirid == "wcurr":
            self.pluscycle()
        elif dirid == "wbest":
            self.plusevo()
        if fname is None:
            idd_str = "" if idd is None else f"_{idd}"
            fname = f"evo-{self.evo_ctr}_cycle-{self.cycle_ctr}{idd_str}"
        fname = f"{dirname}/{fname}"
        if dirid in ["fig", "train", "cpt"]:
            fname = f"{fname}.png"
        elif dirid in ["wcurr", "wbest", "weight", "data"]:
            fname = f"{fname}.pth"
        self.latest[dirid] = fname
        return fname

    def getdirname(self, dirid):
        if dirid not in self.dirs:
            print(f"TrainingFileManager.getdirname: unknown dirid '{dirid}'")
            breakpoint()
        dirname = self.dirs[dirid]
        return dirname

    def getall(self, dirid):
        dirname = self.getdirname(dirid)
        fnames = os.listdir(dirname)
        return [f"{dirname}/{fnames[i]}" for i in range(len(fnames))]

    def getlatest(self, dirid, soft_fail=False):
        if not self.latest[dirid]:
            if soft_fail == True:
                return False
            print(f"TrainingFileManager.getlatest: no entries recoreded")
            breakpoint()
        return self.latest[dirid]
    
    def pluscycle(self):
        self.cycle_ctr += 1
    def plusevo(self):
        self.evo_ctr += 1
        self.cycle_ctr = 0

    def print(self):
        print("--------------------")
        print(f"TrainingFileManager:")
        print(f"  Evolution number: {self.evo_ctr}")
        print(f"  Cycle number    : {self.cycle_ctr}")
        print(f"  File structure  :")
        os.system(f"tree {self.dir}")
        print(f"-------------------")

def main():
    tfm = TrainingFileManager()
    tfm.print()
    t = torch.Tensor([3,4,5])
    fig, ax = plt.subplots()
    ax.plot([2,3], [4,5])
    for evo in range(3):
        for cycle in range(3):
            tfm.save(t, "wcurr")
            tfm.save(t, "wbest")
            tfm.save(fig, "cpt")
            tfm.save(fig, "train")
            tfm.save(fig, "fig")
            tfm.pluscycle()
        tfm.plusevo()
    print(os.system(f"cd {tfm.dir} && tree"))

if __name__ == "__main__":
    main()


