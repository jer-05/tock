import torch 
from torch import nn 
import traceback
import numpy as np
from torchviz import make_dot
from torchinfo import summary

from tock import make, RANKS, DEAL_ORDER, AVERAGE_GAME_LENGTH
from data import GameData
from utils import totorch, getstate


class PolicyNN(nn.Module):
    """
    A network which evaluates the game state and returns a policy head
    and value head
    """
    def __init__(self, nplayers = 2):
        super().__init__()
        self.nplayers = nplayers
        ker1 = 7
        ch1 = 32
        ker2 = 5
        ch2 = 64
        lin1 = 512
        lin2 = 256
        spat_in = nplayers * 16
        spat_out = ch2 * spat_in
        info_in = 2 * (nplayers + 3) + 13 + 5
        p_drop = 0.3

        assert ker1 % 2 == 1
        self.spatial = nn.Sequential(
                # circular padding here, because the board wraps around at the end!
                nn.Conv1d(3 + nplayers, ch1, ker1, padding=ker1//2, padding_mode = 'circular'),
                nn.BatchNorm1d(ch1),
                nn.ReLU(),
                nn.Conv1d(ch1, ch2, ker2, padding=ker2//2),
                nn.BatchNorm1d(ch2),
                nn.ReLU(),
                )
        self.linear1 = nn.Sequential(
                nn.Linear(info_in, lin1),
                nn.ReLU(),
                nn.Dropout(p=p_drop)
                )
        self.linear2 = nn.Sequential(
                nn.Linear(spat_out + lin1, lin2),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
                )
        self.policy_head = nn.Sequential(
                nn.Linear(lin2, 50),
                )
        if self.nplayers == 2:
            self.value_head = nn.Sequential(
                    nn.Linear(lin2, 1),
                    nn.Tanh(),
                    )
        else:
            self.value_head = nn.Sequential(
                    nn.Linear(lin2, self.nplayers),
            )

    def forward(self, field, start, base, card, fold, roundn, player, action_mask):
        # all values are normalized before usage
        round_mod_norm = (roundn % 3) / 3        
        roundn_norm = roundn / (AVERAGE_GAME_LENGTH / (sum(DEAL_ORDER) / len(DEAL_ORDER)))
        player_norm = player / self.nplayers
        ncard = torch.sum(card, dim=1).unsqueeze(1)
        ncard_norm = ncard / max(DEAL_ORDER)
        card_norm = card / self.nplayers
        fold_norm = fold / 3

        # concatenate the linear inputs
        flat = torch.cat((start, base, card_norm, fold_norm, roundn_norm, round_mod_norm, player_norm, ncard_norm), dim=1)

        # compute the spatial part
        spatial = self.spatial(field)

        # compute the linear part
        linear1 = self.linear1(flat)
        in2 = torch.cat((spatial.view(spatial.size(0), -1), linear1), dim=1)
        linear2 = self.linear2(in2)
        policy_head = self.policy_head(linear2)
        masked_logits = policy_head - (1 - action_mask) * 1e5
        value_head = self.value_head(linear2)
        return masked_logits, value_head

    def debug_forward(self, field, start, base, card, fold, roundn, player, action_mask):
        try:
            # print(f"Forwarding:")
            round_mod_norm = (roundn % 3) / 3
            roundn_norm = roundn / (AVERAGE_GAME_LENGTH / (sum(DEAL_ORDER) / len(DEAL_ORDER)))
            player_norm = player / self.nplayers
            ncard = torch.sum(card, dim=1).unsqueeze(1)
            ncard_norm = ncard / max(DEAL_ORDER)
            card_norm = card / self.nplayers
            spatial = self.spatial(field)
            fold_norm = fold / 3

            # print(f"Prepared metadata")

            flat = torch.cat((start, base, card_norm, fold_norm, roundn_norm, round_mod_norm, player_norm, ncard_norm), dim=1)
            # print(f"Flattened metadata")
            names = ["round_mod_norm", "roundn_norm", "player_norm", "ncard", "ncard_norm", "card_norm", "spatial", "fold_norm", "flat"]
            varss = [round_mod_norm, roundn_norm, player_norm, ncard, ncard_norm, card_norm, spatial, fold_norm, flat]
            # for name, var in zip(names, varss):
            #     self.check_var(var, name)
            # assert False
            linear1 = self.linear1(flat)
            # print(f"Executed linear1", flush=True)
            in2 = torch.cat((spatial.view(spatial.size(0), -1), linear1), dim=1)
            linear2 = self.linear2(in2)
            # print(f"Executed linear2")
            policy_head = self.policy_head(linear2)
            # # print(f"Unmasked logits are: {policy_head[0].reshape(13, 4)}")
            # print(f"Got logits")
            masked_logits = policy_head - (1 - action_mask) * 1e5
            # print(f"Masked logits")
            # # print(f"Masked logits are: {masked_logits[0].reshape(13, 4)}")
            value_head = self.value_head(linear2)
            # print(f"Got value head")
            return masked_logits, value_head
        except Exception as e:

            traceback.print_exc()
            print(f"Error: {e}")
            print(f"Showing state dict:")
            for key, value in self.state_dict().items():
                self.check_var(value, key)
            names = ["field", "start", "base", "card", "fold", "roundn", "player"]
            for i, var in enumerate([field, start, base, card, fold, roundn, player]):
                self.check_var(var, names[i])
            breakpoint()

    def check_var(self, value, key):
        def _check_infinite(var, name):
            if not torch.isfinite(var).all():
                print(f"    {name} has infinite/NaN values!!")
                breakpoint()
            else:
                print(f"    {name} has no infinite/NaN values")
        print(f"{key} has:\n"
              f"   shape: {value.shape}\n"
              f"   type: {value.dtype}\n"
              f"   device: {value.device}"
              )
        _check_infinite(value, key)

    def load(self, fname, silent=False):
        if not silent:
            print(f"Loading model weights from {fname}")
        weights = torch.load(fname, map_location=next(self.parameters()).device, weights_only=True)
        self.load_state_dict(weights)

    def save(self, fname):
        print(f"Saving model weights as {fname}")
        torch.save(self.state_dict(), fname)

    def graph_and_summary(self, fname):
        print(f"Saving model graph")
        game = make(self.nplayers)
        obs, rew, done, info = game.reset()
        torchstate = getstate(obs, info, gettorch=True, unsqueeze=True)
        y = self(*torchstate)
        make_dot(y, params=dict(self.named_parameters())).render(fname, format="png")
        print(f"Writing summary")
        summary(self, input_size=tuple(torchstate[i].shape for i in range(len(torchstate))))

def main():
    # # torch.cuda.empty_cache()
    # device = torch.device('cuda')
    # # device = torch.device('cpu')
    # data = GameData()
    # data_fname = "data/TSPEnv_maxdepth-5_return_type-probabilities_store_rng-False_ngames_80_nplayers_4size10074.pth"
    # data.load(data_fname)
    # data.print(5)
    # data.print(7)
    # vram_buffer = torch.empty(1024*1024*512, dtype=torch.float32, device=device)
    # nn = PolicyNN(nplayers=4).to(device)
    # X1, y1 = data[5]
    # X2, y2 = data[7]
    # X = [torch.stack(x).to(device) for x in zip(X1, X2)]
    # dummyX = [torch.randn(X[i].shape, device=device) for i in range(len(X))]
    # print(f"X has len {len(X)}")
    # nn.eval()
    # logits, value_head = nn(*dummyX)
    # print(f"Got logits {logits.shape}")
    # print(logits)
    # print(f"Got v-head {value_head.shape}")
    # print(value_head)
    model = PolicyNN(2)
    model.graph_and_summary("figures/network_graph")



if __name__ == '__main__':
    main()
