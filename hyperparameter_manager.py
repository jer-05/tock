from utils import print_important

class HyperparameterManager:
    def __init__(self, hyperparams, *, ncycle_not_evolved=0):
        self.nworkers = hyperparams['nworkers']
        def _fit_workers(size):
            modded_size = size // self.nworkers * self.nworkers
            if size % self.nworkers != 0:
                modded_size += self.nworkers
            return modded_size
        self.hyperparams = {}
        needs_fitting = ["evo_competition_games", "evo_benchmark_games", "self_play_ngames" ]
        self.mod_factors = {
                "self_play_mcts_maxit": 1.15,
                "self_play_epochs_per_cycle": 1.1,
                "train_epochs": 1.05,
                "sampler_gamma_decay": .995,
        }
        self.non_int_type = ["lr", "self_play_epochs_per_cycle", "train_epochs", "sampler_gamma_decay", "evo_bm_lb_games_ratio"]
        self.max_power = 6
        for key, value in hyperparams.items():
            if key in needs_fitting:
                self.hyperparams[key] = _fit_workers(value)
            else:
                self.hyperparams[key] = value
        max_gamma = self.hyperparams["max_gamma_decay"]
        max_exponent = (self.hyperparams["train_play_rounds_per_cycle"] / self.hyperparams["self_play_epochs_per_cycle"])
        appr_gamma = max_gamma ** (1/max_exponent)
        self.hyperparams["sampler_gamma_decay"] = appr_gamma
        self.switch_cycles = round(self.hyperparams["max_cycle_not_evolved"] / 2 + .01)
        self.cycles_nevo = ncycle_not_evolved

        print(f"Successfully initialized HyperparameterManager")
        print(f"Base Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"    {key}: {value}")

    def __getitem__(self, key, get_base=False):
        if key not in self.mod_factors or get_base:
            return self.hyperparams[key]
        else:
            base_power = self.cycles_nevo
            if self.switch_datagen_model():
                base_power -= self.switch_cycles
            power = max(0, min(base_power, self.max_power))
            factor = self.mod_factors[key] ** power
            modded_val = self.hyperparams[key] * factor
            if key not in self.non_int_type:
                modded_val = int(modded_val)
            return modded_val

    def pluscycle(self):
        self.cycles_nevo += 1

    def plusevolve(self):
        self.cycles_nevo = 0

    def switch_datagen_model(self):
        return self.cycles_nevo >= self.switch_cycles

    def stop_training(self, verbose=True):
        stop = self.cycles_nevo >= self.hyperparams["max_cycle_not_evolved"]
        if stop and verbose:
            print_important(f"Model was not evolved for {self.cycles_nevo} rounds. Halting training.")
        return stop

    def __repr__(self):
        s  = "============================================\n"
        s += "HyperparameterManager:\n"
        s += f"  Cycles not evolved   : {self.cycles_nevo}\n"
        s += f"  Data generation model: {'best' if self.switch_datagen_model() else 'current'}\n"
        s +=  "  Modifiable training parameters:\n"
        for key, value in self.hyperparams.items():
            if key in self.mod_factors:
                modded_val = self[key]
                if key in self.non_int_type:
                    fmt = ".2f"
                else:
                    fmt = ""
                if modded_val != value:
                    s += f"     {key}: {format(value, fmt)} -> {format(modded_val, fmt)}\n"
                else:
                    s += f"     {key}: {format(value, fmt)}\n"
        s += "============================================"
        return s

		
		
		
