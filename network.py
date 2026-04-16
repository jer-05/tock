import torch 
from torch import nn 
import traceback
import numpy as np
import hiddenlayer as hl
from torchviz import make_dot
from torchinfo import summary

from tock import make, RANKS, DEAL_ORDER, AVERAGE_GAME_LENGTH
from data import GameData
from utils import totorch, getstate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                nn.Linear(lin2, 52),
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
        self.softmax = nn.Softmax(dim=-1)
        self.p_loss = nn.CrossEntropyLoss()

        # loss function are different for 2+ players, because then
        # the value head is not a scalar but an array of length nplayers
        # that sums up to 1 (giving the predicted winning ratio for each player)
        if self.nplayers == 2:
            self.v_loss = nn.MSELoss()
        else:
            self.v_loss = nn.CrossEntropyLoss()

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

    def load(self, fname, silent=False, device=None):
        if not silent:
            print(f"Loading model weights from {fname}")
        if device is None:
            device = 'cpu' if not torch.cuda.is_available() else torch.device('cuda')
        weights = torch.load(fname, map_location=device, weights_only=True)
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


class PolicyNN_fully_convolutional(nn.Module):
    def __init__(self, nplayers = 2):
        super().__init__()
        self.nplayers = nplayers
        self.board_len = 16*nplayers
        ker1 = 7
        self.spat_padding = ker1 // 2
        spat_in_size = 13 + 3*(nplayers+1) + 2
        ch1 = 64
        ker2 = 5
        ch2 = 64
        ker3 = 3
        ch3 = 64
        lin_in = ch3 * self.board_len
        lin_out = 256
        spat_in = nplayers * 16
        spat_out = int((spat_in / 2 - ker2 + 1) / 2)
        info_in = 2 * (nplayers + 3) + 13 + 5
        p_drop = 0.3

        assert ker1 % 2 == 1
        self.spatial = nn.Sequential(
                nn.Conv1d(spat_in_size, ch1, ker1, padding=0),
                nn.BatchNorm1d(ch1),
                nn.ReLU(),

                nn.Conv1d(ch1, ch2, ker2, padding=ker2//2),
                nn.BatchNorm1d(ch2),
                nn.ReLU(),

                nn.Conv1d(ch2, ch3, ker3, padding=ker3//2),
                nn.BatchNorm1d(ch3),
                )
        self.linear = nn.Sequential(
                nn.Linear(lin_in, lin_out),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
        )

        self.policy_logits = nn.Sequential(
                nn.Linear(lin_out, 13),
                )
        self.value_head = nn.Sequential(
                nn.Linear(lin_out, 1),
                nn.Tanh(),
                )
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=-1)
        self.p_loss = nn.CrossEntropyLoss()
        self.v_loss = nn.MSELoss()

    def forward(self, field, start, base, card, fold, roundn, player, action_mask):
        try:
            roundn_norm = roundn / (AVERAGE_GAME_LENGTH / (sum(DEAL_ORDER) / len(DEAL_ORDER)))
            ncard = torch.sum(card, dim=1).unsqueeze(1)
            card_norm = card / self.nplayers
            fold_norm = fold / 3

            # print("Padding field!")
            padded_field = torch.cat((field[:, :, -self.spat_padding:], field, field[:, :, :self.spat_padding]), dim=-1)

            own_pawns = padded_field[:, :4]
            B = own_pawns.shape[0]
            pad_bl = self.board_len + 2*self.spat_padding
            all_own_pawns = torch.sum(own_pawns, dim=1).view(B, 1, pad_bl)
            spatial_cards = card.unsqueeze(2).expand(B, 13, pad_bl) / self.nplayers
            spatial_round = roundn_norm.unsqueeze(2).expand(B, 1, pad_bl)
            spatial_fold = fold.unsqueeze(2).expand(B, 1, pad_bl)

            # print("Handling bases")

            own_start = start[:, :4]
            own_base = base[:, :4]
            sum_own_start = torch.sum(own_start, dim=1).view(B, 1) / 3
            sum_own_base = torch.sum(own_base, dim=1).view(B, 1) / 3

            # print("abs")
            all_base_start = torch.cat((sum_own_base, base[:, 4:]/3, sum_own_start, start[:, 4:]/3), dim=1)
            abs_expanded = all_base_start.view(B, -1, 1).expand(B, -1, pad_bl)




            own_pawns_expanded = own_pawns.reshape(B*4, 1, pad_bl)
            own_start_expanded = start[:, :4].unsqueeze(2).expand(B, 4, pad_bl).reshape(B*4, 1, pad_bl)
            own_base_expanded = base[:, :4].unsqueeze(2).expand(B, 4, pad_bl).reshape(B*4, 1, pad_bl)
            # print("Catting own_data!")
            own_data = torch.cat((own_pawns_expanded, own_start_expanded, own_base_expanded), dim=1)
            # print("Let's get the spatial_in")
            all_spatial_in = torch.cat((all_own_pawns, padded_field[:, 4:], spatial_cards, spatial_round, spatial_fold, abs_expanded ), dim=1)


            repeated_spatial_in = torch.repeat_interleave(all_spatial_in, repeats=4, dim=0)

            # print("Concatting own_data, spatial in ")
            spatial_in = torch.cat((own_data, repeated_spatial_in), dim=1)
            # print("Forwarding spatial")

            spatial_out = self.spatial(spatial_in)

            lin_in = torch.flatten(spatial_out, start_dim=1)
            # print("Getting the lin_out now")
            lin_out = self.linear(lin_in)

            # print("Getting policy")
            four_policy_logits = self.policy_logits(lin_out)
            # print("Getting value")
            four_value_head = self.value_head(lin_out)
            # logits, value should be shape (BX4) X 13, (BX4) X 1, respectively

            policy_logits = four_policy_logits.view(B, 52)
            value_head = four_value_head.view(B, 4)

            trans_policy_logits = four_policy_logits.view(B, 4, 13).transpose(0, 1)
            masked_logits = trans_policy_logits - (1 - action_mask.reshape(B, 4, 13).transpose(0, 1)) * 1e5
            lse_weights = torch.logsumexp(masked_logits.detach(), dim=2)
            weights = torch.softmax(lse_weights, dim=0)
            # print("Reweighing value heads")
            value_head = value_head * weights.T

            return masked_logits.transpose(0,1).reshape(B, 52), torch.sum(value_head, dim=1, keepdim=True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Field: {field.shape}")
            print(f"Start: {start.shape}")
            print(f"Base: {base.shape}")
            print(f"Card: {card.shape}")
            print(f"Fold: {fold.shape}")
            print(f"Roundn: {roundn.shape}")
            print(f"Player: {player.shape}")
            breakpoint()

    def load(self, fname):
        print(f"Loading model weights from {fname}")
        weights = torch.load(fname)
        self.load_state_dict(weights)

    def save(self, fname):
        print(f"Saving model weights as {fname}")
        torch.save(self.state_dict(), fname)

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
