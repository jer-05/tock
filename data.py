import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import traceback
import os

from tock import RANKS, SUITS, DEAL_ORDER
from utils import field_start_base_to_obs, totorch, checkfn

class GameData(Dataset):
    """
    X: tuple (field, base, start, card, fold, roundn, player)
    y: tuple (y_prob, y_val)
    """
    def __init__(self, buffer_size = 10000000):
        super().__init__()
        self.X = []
        self.y = []
        self.buffer_size = buffer_size

    def __getitem__(self, idx, gettorch=True):
        assert len(self) <= self.buffer_size
        sparse_field, start, base, card, fold, roundn, player, sparse_mask = self.X[idx]
        nplayers = len(start) - 3
        field = np.zeros((nplayers + 3, nplayers * 16))
        field[sparse_field] = 1
        action_mask = np.zeros(52)
        action_mask[sparse_mask] = 1
        sparse_prob_idx, sparse_prob, y_val = self.y[idx]
        if nplayers != 2:
            tmp = y_val
            y_val = np.zeros(nplayers)
            y_val[tmp] = 1
        dtype = torch.float32
        y_prob = np.zeros(52)
        y_prob[sparse_prob_idx] = sparse_prob
        if gettorch:
            return totorch((field, start, base, card, fold, roundn, player, action_mask)), totorch((y_prob, y_val))
        return (field, start, base, card, fold, roundn, player, action_mask), (y_prob, y_val)

    def append(self, field, start, base, card, fold, roundn, player, action_mask, y_prob, y_val):
        nplayers = len(field) - 3
        if len(self.X) >= self.buffer_size:
            self.X.pop(0)
            self.y.pop(0)
        try:
            sparse_field = field.nonzero()
            sparse_field = tuple(sparse_field[i].astype('uint8') for i in range(2))
            sparse_mask = action_mask.nonzero()[0].astype('uint8')
            sparse_prob_idx = y_prob.nonzero()[0].astype('uint8')
            sparse_prob = y_prob[sparse_prob_idx].astype('float32')
            if nplayers != 2:
                y_val = list(y_val).index(1)
        except:
            traceback.print_exc()
            breakpoint()
        self.X.append((sparse_field, start, base, card, fold, roundn, player, sparse_mask))
        self.y.append((sparse_prob_idx, sparse_prob, y_val))

    def print(self, idx):
        print(f"--------------------------")
        print(f"Printing data sample {idx}")
        sparse_field, start, base, card, fold, roundn, player, sparse_mask = self.X[idx]
        nplayers = len(start) - 3
        field = np.zeros((nplayers + 3, nplayers * 16), dtype='uint8')
        field[sparse_field] = 1
        action_mask = np.zeros(52, dtype='uint8')
        action_mask[sparse_mask] = 1
        sparse_prob_idx, sparse_prob, y_val = self.y[idx]
        if nplayers != 2:
            tmp = y_val
            y_val = np.zeros(nplayers)
            y_val[tmp] = 1
        dtype = torch.float32
        y_prob = np.zeros(52)
        y_prob[sparse_prob_idx] = sparse_prob
        pawn_locs = field_start_base_to_obs(field, start, base)
        print(f"Player: {player}")
        print(f"Round: {roundn}")
        if nplayers != 2:
            y_val = list(y_val)
            y_val_str = ""
            for val in y_val:
                y_val_str += f"{val:.3f}, "
            y_val_str = y_val_str[:-2]
            print(f"Value Target: {y_val_str}")
        else:
            print(f"Value Target: {y_val:.3f}")
        print(f"Pawn Locations:\n"
              f"    Current player: {pawn_locs[0]}")
        for i in range(1, len(field) - 3):
            print(f"    Opponent {i}    : {pawn_locs[i]}")
        card_str = ""
        for i in range(13):
            ncard = int(card[i])
            for j in range(ncard):
                card_str += str(RANKS[i]) + ", "
        card_str = card_str[:-2]
        ncards = int(sum(card))
        print(f"Cards: {card_str}  (total {ncards})")
        prob_str = ""
        print(f"Probabilities:")
        for i, prob in enumerate(y_prob):
            if prob != 0:
                card, pawn = i // 4, i % 4
                card_symbol = RANKS[card]
                prob_str += f"    ({card_symbol}, {pawn}) =>  p={prob:.3f}\n"
            elif action_mask[i] != 0:
                print("Error: action mask does not correspond with the given probabilities!")
                breakpoint()
        prob_str = prob_str[:-1]
        print(prob_str)
        print(f"Action mask:")
        print(action_mask.reshape(13, 4))
        assert fold in [0, 1, 3]
        if fold == 0:
            print(f"Previous player folded")
        elif fold == 3:
            print(f"New round started")
        elif fold == 1:
            print(f"Previous player played a card")
        print(f"--------------------------")

    def save(self, fname, bak=True, chunk_size = 400000,*, test=False):
        fname = fname[:-4] + f"_size{len(self)}"
        if bak:
            checkfn(fname)
        print(f"Saving data to '{fname}'")
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
            blob_fname = f"{fname}_blob{blobid}.pth"
            print(f"Saving blob {blobid} as {blob_fname} with data [{start}:{stop}]")
            torch.save(state, blob_fname)
            blobid += 1
            sample += chunk_size
        if test:
            xcp = copy.deepcopy(self.X)
            ycp = copy.deepcopy(self.y)
            self.load_blobs(fname)
            if len(self.X) == 2*len(xcp) and len(self.y) == 2*len(ycp):
                print("success")

    def load_blobs(self, fname):
        i=0
        while True:
            blobfname = fname + f"_blob{i}.pth"
            if not os.path.isfile(blobfname):
                break
            blobdata = torch.load(blobfname, weights_only=False)
            self.X += blobdata['X']
            self.y += blobdata['y']
            print(f"Loaded blob {i} with fname {blobfname} with {len(blobdata['X'])} samples")
            i += 1
        self.buffer_size = max(len(self), self.buffer_size)
        print(f"Loaded {i} blobs. New size is {len(self)}")


    def cut_buffer_size(self, size):
        self.buffer_size = size
        if len(self) > size:
            self.X = self.X[len(self.X) - size:]
            self.y = self.y[len(self.y) - size:]
        print(f"Cut buffer size to {self.buffer_size}")

    def load(self, fname=None, data=None, append=False, load_from_data=False):
        if not append:
            if not load_from_data:
                print(f"Loading data from '{fname}'")
                state = torch.load(fname, weights_only = False)
                self.X = state['X']
                self.y = state['y']
                self.buffer_size = state['buffer_size']
            else:
                assert data is not None
                print(f"Loading data from data object")
                state = copy.deepcopy(data)
                assert len(self) == 0, "Dataset non-empty, set append=True to append data"
                self.X = state.X
                self.y = state.y
                self.buffer_size = state.buffer_size
        elif load_from_data:
            print(f"Appending data from data object with {len(data)} samples to current database of {len(self)} samples")
            state = copy.deepcopy(data)
            assert len(state.X) <= self.buffer_size
            newsize = len(self) + len(state)
            if newsize > self.buffer_size:
                self.X = self.X[newsize-self.buffer_size:]
                self.y = self.y[newsize-self.buffer_size:]
                print(f"Cutting first {newsize - self.buffer_size} samples!")
            self.X += state.X
            self.y += state.y
        else:
            print(f"load: Operation not supported")
            breakpoint()
        print(f"Now {len(self)} samples in database")


    def show_random(self, n=10):
        for i in range(n):
            idx = random.randint(0, len(self) - 1)
            self.print(idx)



    def __len__(self):
        return len(self.X)

def main():
    data = GameData(buffer_size=50)
    data.load("data/TSPEnv_maxdepth-5_return_type-probabilities_store_rng-False_ngames_80size10003.pth")
    data.show_random()
    # data2 = GameData(buffer_size=50)
    # data2.load("data/TSPEnv_maxdepth-8_return_type-probabilities_store_rng-False_TSPEnv_maxdepth-8_return_type-probabilities_store_rng-False.pth.bak")
    breakpoint()


if __name__ == '__main__':
    main()

