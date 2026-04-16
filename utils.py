from random import choice 
import numpy as np 
import torch 
from torch import nn 
import os
import traceback
import sys

from tock import make, PLACES_PER_SEGMENT, RANKS, DEAL_ORDER, AVERAGE_GAME_LENGTH

device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

def getstate(obs, info, gettorch = False, unsqueeze = True): 
    """
    Get the game state representation
    """
    current_player = info['player'] 
    nplayer = len(obs) 
    nfields = nplayer * PLACES_PER_SEGMENT 
    offsets = [PLACES_PER_SEGMENT * ((i - current_player) % nplayer)\
            for i in range(nplayer)]
    field = np.zeros((nplayer + 3, nfields))
    start = np.zeros(nplayer + 3)
    base = np.zeros(nplayer + 3)

    # compute absolute locations relative to the current player
    # the representation is a one-hot encoding
    for player in range(nplayer):
        shifted_player = (player - current_player) % nplayer
        for pawn in range(4):
            if not shifted_player: # player is current player
                channel_offset = pawn
            else:
                channel_offset = 3
            loc = obs[player][pawn]
            if loc == 0:
                start[shifted_player + channel_offset] += 1
            elif loc == nfields + 1:
                base[shifted_player + channel_offset] += 1
            else:
                abs_loc = (loc - 1 + offsets[player]) % nfields
                field[shifted_player + channel_offset][abs_loc] = 1 

    # prepare other relevant information, such as cards, round and action mask
    cards = np.zeros(13)
    for card in info['cards']:
        unique_card = card % 13
        cards[unique_card] += 1
    ncards = int(sum(cards))
    roundn = info['round'] - 1
    if ncards == DEAL_ORDER[roundn % 3]:
        folded = 3
    elif info['action']:
        folded = 1
    else:
        folded = 0
    action_mask = np.zeros(52)
    for action in info['space']:
        cardidx, pawn = action
        card = info['cards'][cardidx] % 13
        action_number = 4 * card + pawn
        action_mask[action_number] = 1
    if not gettorch:
        return field, start, base, cards, folded, roundn, current_player, action_mask
    elif gettorch:
        return totorch((field, start, base, cards, folded, roundn, current_player, action_mask), unsqueeze=unsqueeze)

         
def clean_fname(fname):
    return fname.replace(":", "-").replace("{", "_").replace("}","_").replace("'","").replace('"',"").replace('/','-').replace(" ","")

def remove_dirs(fname):
    if "." in fname:
        fname = fname[:fname.index(".")]
    if "/" not in fname:
        return fname
    inverse_fname = fname[::-1]
    slash_index = inverse_fname.index("/")
    inverse_fname = inverse_fname[:slash_index]
    return clean_fname(inverse_fname[::-1])

def clear_lines(nlines):
    sys.stdout.write(f"\033[{nlines}F")
    print("\n"*(nlines-1))
    sys.stdout.write(f"\033[{nlines}F")


def checkfn(fname):
    if os.path.isfile(fname):
        new_fname = f"{fname}.bak"
        os.system(f"mv '{fname}' '{new_fname}'")
        print(f"File '{fname}' exists, moved old to '{new_fname}'")

def get_idx(card, pawn, cards):
    return (cards[card] % 13)* 4 + pawn

def print_evals_and_info(info, evals, gamestr, return_string=False, twoD_eval=False, hyphens=True):
    if not twoD_eval:
        assert len(evals) == len(info['space'])
        eval_order = np.argsort(evals)[::-1]
        ordered_evals = evals[eval_order]
    else:
        assert len(evals[0]) == len(info['space'])
        eval_order = np.argsort(evals[0])[::-1]
        ordered_evals = evals[:, eval_order]
    current_player = info['player']
    nactions = len(info['space'])
    action_ranking = [info['space'][eval_order[i]] for i in range(nactions)]
    action_str = ""
    for i in range(nactions):
        action = get_action_string(action_ranking[i], info['cards'])
        if not twoD_eval:
            action_str += f"    {action}  => {ordered_evals[i]:.3f}\n"
        else:
            action_str += f"    {action}  => {ordered_evals[0][i]:.3f},  {ordered_evals[1][i]:.3f}\n"

    action_str = action_str[:-1]
    if return_string:
        return gamestr + action_str
    if hyphens:
        print("------------------")
    print(gamestr)
    print(action_str)
    if hyphens:
        print("------------------")

def get_action_string(action, cards):
    card_idx, pawn = action
    card = cards[card_idx] % 13
    card_name = RANKS[card]
    card_str = f"({card_name}, {pawn})"
    return card_str

def get_card_str(cards):
    card_str = "Cards: "
    for card in cards:
        unique_card = card % 13
        card_name = RANKS[unique_card]
        card_str += f"{card_name}, "
    card_str = card_str[:-2]
    card_str += "\n"
    return card_str

def get_obs_str(obs, player):
    rotated_obs = [obs[i - player] for i in range(len(obs))]
    s = ""
    for i, obsi in enumerate(rotated_obs):
        s += f"P{i}: {obsi}\n"
    return s

    
    
def get_unique_actions(action_space, cards):
    unique_mask = np.ones(len(cards))
    visited_cards = []
    for card_idx, card in enumerate(cards):
        card = card % 13
        if card in visited_cards:
            unique_mask[card_idx] = 0
        else:
            visited_cards.append(card)
    unique_space = []
    for action in action_space:
        card_idx = action[0]
        if not unique_mask[card_idx]:
            continue
        unique_space.append(action)
    return unique_space

def deuniqueify_prob(action_space, cards, unique_prob):
    unique_actions = get_unique_actions(action_space, cards)
    non_unique_prob = np.zeros(len(action_space))
    for i, unique_action in enumerate(unique_actions):
        t_unique_action = (cards[unique_action[0]] % 13, unique_action[1])
        saved_indices = []
        for j, action in enumerate(action_space):
            t_action = (cards[action[0]] % 13, action[1])
            if t_unique_action == t_action:
                non_unique_prob[j] = unique_prob[i]
                saved_indices.append(j)
        for idx in saved_indices:
            non_unique_prob[idx] /= len(saved_indices)
    return non_unique_prob

def get_action_prob(prob, info, return_mask=False):
    try:
        mask = np.zeros(52)
        idx_order = []
        action_prob = np.zeros(len(info['space']))
        for i, action in enumerate(info['space']):
            idx = get_idx(*action, info['cards'])
            action_prob[i] = prob[idx]
            mask[idx] = 1
            idx_order.append(idx)
        idx_order = np.array(idx_order)
        action_prob = action_prob / np.sum(action_prob)
        one = np.sum(action_prob)
        if not np.isclose(one, 1):
            print(f"Error: Sum of reverted prob. should be 1 but is {one}")
            print(f"prob: {prob}")
            print(f"info: {info}")
            breakpoint()
        if return_mask:
            assert sum(mask) == len(action_prob)
            idx_order = np.argsort(idx_order)
            return action_prob, mask.astype(bool), np.argsort(idx_order)
        return action_prob
    except:
        traceback.print_exc()
        breakpoint()

def get_action_from_idx(idx, cards):
    assert (idx >= 0 and idx < 52)
    card = idx // 4
    cardidx = -1
    for offset in [0, 13, 26, 39]:
        offsetted_card = card + offset
        if offsetted_card in cards:
            cardidx = cards.index(offsetted_card)
            break
    pawn = idx % 4
    card_name = RANKS[card]
    return (cardidx, pawn), (card_name, pawn)

def policy_to_actions(policy_head, cards):
    list_order = np.argsort(policy_head)[::-1]
    actions = []
    print_actions = []
    for i in range(52):
        action, print_action = get_action_from_idx(list_order[i], cards)
        actions.append(action)
        print_actions.append(print_action)
    return actions, print_actions


def revert_action_prob(action_prob, info):
    assert len(action_prob) == len(info['space']), f"Incorrect amount of probabilities supplied, {len(action_prob)} vs. {len(info['space'])}"
    prob = np.zeros(13*4)
    for i, action in enumerate(info['space']):
        prob[get_idx(*action, info['cards'])] += action_prob[i]
    one = np.sum(prob)
    if not np.isclose(one, 1):
        print(f"Error: Sum of reverted prob. should be 1 but is {one}")
        breakpoint()
    return prob



class PolicyNN(nn.Module):
    def __init__(self, nplayers = 2):
        super().__init__()
        self.nplayers = nplayers
        ker1 = 7
        ch1 = 64
        ker2 = 3
        ch2 = 128
        lin1 = 256
        lin2 = 300
        spat_in = nplayers * 16
        spat_out = int((spat_in / 2 - ker2 + 1) / 2)
        info_in = 2 * (nplayers + 3) + 13 + 5
        p_drop = 0.3

        assert ker1 % 2 == 1
        self.spatial = nn.Sequential(
                nn.Conv1d(3 + nplayers, ch1, ker1, padding=ker1//2, padding_mode = 'circular'),
                nn.BatchNorm1d(ch1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(ch1, ch2, ker2),
                nn.BatchNorm1d(ch2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(p=p_drop),
                )
        self.linear1 = nn.Sequential(
                nn.Linear(info_in, lin1),
                nn.ReLU(),
                nn.Dropout(p=p_drop)
                )
        self.linear2 = nn.Sequential(
                nn.Linear(ch2 * spat_out + lin1, lin2),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
                )
        self.policy_head = nn.Sequential(
                nn.Linear(lin2, 52),
                )
        self.value_head = nn.Sequential(
                nn.Linear(lin2, 1),
                nn.Tanh(),
                )
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=-1)
        self.p_loss = nn.CrossEntropyLoss()
        self.v_loss = nn.MSELoss()
    def forward(self, field, start, base, card, fold, roundn, player):
        try:
            round_mod_norm = (roundn % 3) / 3
            roundn_norm = roundn / (AVERAGE_GAME_LENGTH / (sum(DEAL_ORDER) / len(DEAL_ORDER)))
            player_norm = player / self.nplayers
            ncard = torch.sum(card, dim=1).unsqueeze(1)
            ncard_norm = ncard / max(DEAL_ORDER)
            card_norm = card / self.nplayers
            spatial = self.spatial(field)
            fold_norm = fold / 3

            flat = torch.cat((start, base, card_norm, fold_norm, roundn_norm, round_mod_norm, player_norm, ncard_norm), dim=1)
            linear1 = self.linear1(flat)
            in2 = torch.cat((spatial.view(spatial.size(0), -1), linear1), dim=1)
            linear2 = self.dropout(self.linear2(self.dropout(in2)))
            policy_head = self.policy_head(linear2)
            value_head = self.value_head(linear2)
            return policy_head, value_head
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


def totorch(arrs, dtype = torch.float32, unsqueeze=False, device='cpu'):
    converted_arrs = []
    for arr in arrs:
        if type(arr) == np.ndarray:
            converted_arrs.append(torch.from_numpy(arr).to(dtype).to(device))
        elif np.isscalar(arr):
            converted_arrs.append(torch.Tensor([arr]).to(dtype).to(device))
    if unsqueeze:
        converted_arrs = [tensor.unsqueeze(0) for tensor in converted_arrs]
    return converted_arrs

# def totf(arrs, dtype=tf.float32, unsqueeze=False):
#     converted_arrs = []
#
#     for arr in arrs:
#         # Check if it's a numpy array or already a tensor
#         if isinstance(arr, (np.ndarray, list)):
#             tensor = tf.convert_to_tensor(arr)
#             converted_arrs.append(tf.cast(tensor, dtype))
#
#         elif np.isscalar(arr):
#             # Create a 1D tensor from a scalar
#             tensor = tf.constant([arr])
#             converted_arrs.append(tf.cast(tensor, dtype))
#
#     if unsqueeze:
#         # tf.expand_dims(x, 0) is the equivalent of torch.unsqueeze(x, 0)
#         converted_arrs = [tf.expand_dims(tensor, axis=0) for tensor in converted_arrs]
#
#     return converted_arrs

def field_start_base_to_obs(field, start, base):
    nplayer = len(field) - 3
    pawn_locs = np.zeros((nplayer, 4))
    board_len = nplayer * 16
    for i, (subfield, substart, subbase) in enumerate(zip(field, start, base)):
        if i < 4:
            if substart == 1:
                pawn_locs[0, i] = 0
            elif subbase == 1:
                pawn_locs[0, i] = board_len + 1
            else:
                idx = np.arange(len(subfield))[subfield == 1][0]
                pawn_locs[0, i] = idx + 1
        else:
            idx = 0
            player = i - 3
            player_offset = 16 * player
            while substart > 0:
                pawn_locs[player, idx] = 0
                idx += 1
                substart -= 1
            while subbase > 0:
                pawn_locs[player, idx] = board_len + 1
                idx += 1
                subbase -= 1
            field_locs = np.arange(len(subfield))[subfield == 1]
            for field_loc in field_locs:
                pawn_locs[player, idx] = (field_loc - player_offset) % board_len+ 1
                idx += 1
    return pawn_locs



def test_representation():
    game = make(2)
    obs, rew, done, info = game.reset()
    nw = PolicyNN()
    while not done:
        print(f"\n--------")
        print(f"Player is {info['player']}")
        print(game)
        field, start, base, card = getstate(obs, info)
                # print(f"Field: {field}")
        # print(f"Start: {start}")
        # print(f"Base: {base}")
        pawn_locs = field_start_base_to_obs(field, start, base)
        print(f"(reverse-engineered) obs:\n"
              f"Current player: {pawn_locs[0]}")
        for i in range(1, len(obs)):
            print(f"Opponent {i}: {pawn_locs[i]}")
        card_str = ""
        for i in range(13):
            ncard = int(card[i])
            for j in range(ncard):
                card_str += str(RANKS[i]) + " "
        print(f"Cards: {card_str}")
        (field, start, base, cards) = totorch((field, start, base, card))
        field = field.view(1, *field.shape)
        start= start.view(1, *start.shape)
        base = base.view(1, *base.shape)
        cards = cards.view(1, *cards.shape)
        
        policy_head, value_head = nw(field, start, base, cards)
        print(f"policy_head: {policy_head}")
        action_prob = get_action_prob(policy_head[0].detach().numpy(), info)
        print(f"action prob: {action_prob}")
        reverted = revert_action_prob(action_prob, info)
        one = np.sum(reverted)
        if not np.isclose(1, one):
            print(f"Error: reverted action prob. do NOT sum to 1 but to {one}")
            breakpoint()
        print(f"reverted action prob: {reverted}")
        print(f"value_head: {value_head}")
        action = choice(info['space'])
        print(f"Chose action (card, pawn) = {action}")
        obs, rew, done, info = game.step(action)


def main():
    test_representation()
    
if __name__ == '__main__':
    main()
