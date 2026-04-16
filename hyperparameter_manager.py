class HyperparameterManager:
    def __init__(self, hyperparams):
        self.nworkers = hyperparams['nworkers']
        def _fit_workers(size):
            return size // self.nworkers * self.nworkers
        self.hyperparams = {}
        needs_fitting = ["evo_competition_games", "evo_benchmark_games", "self_play_ngames"]
        self.mod_factors = {
                "self_play_ngames": 1.1,
                "self_play_mcts_maxit": 1.3,
                "train_steps": 1.15,
                "replay_buffer_size": .85,
                "lr": .8,
        }
        self.rel_to_switch_type = ["self_play_mcts_maxit", "lr"]
        self.non_int_type = ["lr"]
        self.max_power = 6
        for key, value in hyperparams.items():
            if key in needs_fitting:
                self.hyperparams[key] = _fit_workers(value)
            else:
                self.hyperparams[key] = value
        self.ncycle_ne_switch_datagen_to_best = self.hyperparams["max_cycle_not_evolved"] // 2 + 1
        self.ncycle_not_evolved = 0

        print(f"Successfully initialized HyperparameterManager")
        print(f"Base Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"    {key}: {value}")

    def plusnotevolved(self):
        self.ncycle_not_evolved += 1

    def evolved(self):
        self.ncycle_not_evolved = 0

    def __getitem__(self, key):
        if key not in self.mod_factors:
            return self.hyperparams[key]
        else:
            power = self.ncycle_not_evolved
            if key in self.rel_to_switch_type:
                if self.switch_datagen_model():
                    power -= self.ncycle_ne_switch_datagen_to_best
            power = min(power, self.max_power)
            factor = self.mod_factors[key] ** power
            modded_val = self.hyperparams[key] * factor
            if key not in self.non_int_type:
                modded_val = int(modded_val)
            return modded_val

    def switch_datagen_model(self, verbose=False):
        switch =  self.ncycle_not_evolved - self.ncycle_ne_switch_datagen_to_best >= 0
        if verbose and switch:
            print(f"Model did not improve for {self.ncycle_not_evolved} self-play rounds, so using best model for data generation")
        return switch


    def stop_training(self):
        stop = self.ncycle_not_evolved >= self.hyperparams["max_cycle_not_evolved"]
        if stop:
            print(f"Model was not evolved for {self.ncycle_not_evolved} rounds. Halting training.")
        return stop

    def print_evo_state(self):
        print(f"Last evolution was {self.ncycle_not_evolved} cycles ago")

    def __repr__(self):
        s = ""
        s += f"Did not evolve for {self.ncycle_not_evolved} cycles\n"
        s += "Modified training parameters:\n"
        for key, value in self.hyperparams.items():
            if key in self.mod_factors:
                modded_val = self[key]
                if modded_val != value:
                    s += f"    {key}: {value} -> {modded_val}\n"
                else:
                    s += f"    {key}: {value}\n"
        return s[:-1]

		
		
		
