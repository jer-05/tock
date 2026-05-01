from random import choice 
import numpy as np 
import torch 
from torch import nn 
import torch.onnx
import onnxruntime as ort
import time
import os
import traceback
import sys
from contextlib import redirect_stderr
import multiprocessing
from tqdm import tqdm


from tock import make, PLACES_PER_SEGMENT, RANKS, DEAL_ORDER, AVERAGE_GAME_LENGTH
from fasttock import FastTockGame, CARD_TO_REPR, action_tuple_to_action_number, tupled_action_space_to_action_mask


device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

def getstate(unsorted_obs, info, gettorch = False, unsqueeze = True, getdict = False, fastgame_type=True): 
    """
    Get the game state representation
    """
    current_player = info['player'] 
    # print("unsorted: ", unsorted_obs)
    if fastgame_type:
        obs = unsorted_obs
    else:
        obs = [sorted(unsorted_obs[i]) for i in range(len(unsorted_obs))]
    # print("sorted: ", obs)
    nplayer = len(obs) 
    for player in range(nplayer):
        old = -np.inf
        for ob in obs[player]:
            if ob < old:
                print(f"getstate: obs should be supplied in the ordered FastTockGame format")
                breakpoint()
            old = ob
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

    # print(f"Saved locs:")
    # print(field_start_base_to_obs(field, start, base))

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
    if fastgame_type:
        action_mask = info['space']
    else:
        action_mask = tupled_action_space_to_action_mask(info['space'], unsorted_obs[current_player], info['cards'])
    if getdict:
        state = {}
        state['field'] = field[None, :].astype(np.float32)
        state['start'] = start[None, :].astype(np.float32)
        state['base'] = base[None, :].astype(np.float32)
        state['card'] = cards[None, :].astype(np.float32)
        state['fold'] = np.array([[folded]], dtype=np.float32)
        state['roundn'] = np.array([[roundn]], dtype=np.float32)
        state['player'] = np.array([[current_player]], dtype=np.float32)
        state['action_mask'] = action_mask[None, :].astype(np.float32)
        return state
    if not gettorch:
        return field, start, base, cards, folded, roundn, current_player, action_mask
    elif gettorch:
        return totorch((field, start, base, cards, folded, roundn, current_player, action_mask), unsqueeze=unsqueeze)

def print_block(s):
    slen = len(s)
    print("=" * (slen + 4))
    print(f"| {s} |")
    print("=" * (slen + 4))

def print_v(s):
    slen = len(s)
    print("v" * (slen + 4))
    print(f"| {s} |")
    print("^" * (slen + 4))

def print_important(s):
    slen = len(s)
    print(f"[!]: {s}")


def check_abortion(stopevent, reset=False, extra_stop_event=None):
    start = time.time()
    tqdm.write(f"Put 'stop' in stopevent.txt to abort")
    if reset:
        with open("stopevent.txt", "w") as f:
            f.write('running')
    while not stopevent.is_set():
        try:
            with open("stopevent.txt", "r") as f:
                if f.read().strip() == "stop":
                    tqdm.write(f"\n[!]: Abort signal received from file!")
                    if extra_stop_event is not None:
                        extra_stop_event.set()
                    stopevent.set()
        except Exception as e:
            pass
        # print(f"Abortion checker with pid {os.getpid()} is still alive, running for {time.time() - start:.0f} seconds")
        time.sleep(2)
    if reset:
        with open("stopevent.txt", "w") as f:
            f.write('stopped')

ort_options = ort.SessionOptions()
ort_options.intra_op_num_threads = 1
ort_options.inter_op_num_threads = 1

ONNX_WEIGHTS_DIR = "weights/onnx"

def get_ort_inference_session(model, mfname, pid, currently_exporting_arr, export_lock, exported_models_cache, verbose=False):
    session_hash = tuple([mfname, os.path.getmtime(mfname), pid])
    export_hash = tuple([mfname, os.path.getmtime(mfname)])
    c = False
    while True:
        with export_lock:
            if export_hash not in currently_exporting_arr:
                currently_exporting_arr[export_hash] = True
                break
        if verbose:
            if c == False:
                print(f"[{pid}]: waiting for export ...")
                c = True
        time.sleep(.1)

    with export_lock:
        cached = export_hash in exported_models_cache
            
    if cached:
        fname = exported_models_cache[export_hash]
        if verbose:
            print(f"[{pid}]: Retrieving exported model from cache")
    else:
        if verbose:
            print(f"[{pid}]: Export hash {export_hash} not yet in cache ... exporting ...")
        model.eval()
        nplayers = model.nplayers
        game = FastTockGame(nplayers)
        obs, done, rew, info = game._get_gamestate()
        state = getstate(obs, info, gettorch=True, unsqueeze=True)
        i = 0
        while True:
            fname = f"{ONNX_WEIGHTS_DIR}/model_pid{pid}_i{i}.onnx"
            if not os.path.isfile(fname):
                if verbose:
                    print(f"[{pid}]: Exporting ONNX model from fname {mfname}!")
                with redirect_stderr(open("/dev/null", "w")):
                    torch.onnx.export(model, tuple(state), fname, verbose=False, input_names=["field", "start", "base", "card", "fold", "roundn", "player", "action_mask"])
                with export_lock:
                    if verbose:
                        print(f"[{pid}]: Saving to exported models cache w/ hash {export_hash}")
                    exported_models_cache[export_hash] = fname
                break
            i += 1
    sess = ort.InferenceSession(fname, providers=['CPUExecutionProvider'], sess_options = ort_options)
    # print(f"Cached ONNX Runtime session successfully")
    with export_lock:
        currently_exporting_arr.pop(export_hash)
    return sess

import random
def clean_fname(fname):
    cleant_fname = fname.replace(":", "-").replace("{", "_").replace("}","_").replace("'","").replace('"',"").replace('/','-').replace(" ","").replace(",","").replace("?","")
    need_remove_ratio = 1 - 100/len(cleant_fname)
    i = 0
    while True:
        if i >= len(cleant_fname) - 5:
            i = 0
        if len(cleant_fname) < 100:
            break
        cleant_fname = cleant_fname.replace("-","").replace("_","")
        c = cleant_fname[i]
        if c.isalpha() and random.random() < need_remove_ratio:
            if i == len(cleant_fname) -1:
                cleant_fname = cleant_fname[:i]
            else:
                cleant_fname = cleant_fname[:i] + cleant_fname[i+1:]
        else:
            i += 1
    return cleant_fname

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

def get_action_string(action, cards):
    card_idx, pawn = action
    card = cards[card_idx] % 13
    card_name = CARD_TO_REPR[card]
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
        s += f"P{i}: {', '.join(f'{ob:.0f}' for ob in obsi)}\n"
    return s

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
    pawn_locs = list(pawn_locs)
    return [sorted(pawn_locs[i]) for i in range(nplayer)]



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
    s = "hello guys"
    print_block(s)
    print_v(s)
    print_important(s)
    # test_representation()
    
if __name__ == '__main__':
    main()
