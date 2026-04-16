import torch
from time import time
import os
import torch.multiprocessing as mp
import multiprocessing

from network import PolicyNN
from utils import getstate
from tock import make
from contextlib import redirect_stdout

def get_gpu_manager(*, worker_batch_size, nworkers, model_path, info_file="gpu_manager_info.txt"):
    assert torch.cuda.is_available(), "Cannot get GPU manager; cuda not available"
    mp.set_start_method("spawn", force=True)
    gpu_manager = GPUManager(model_path, max_batch=worker_batch_size, nworkers=nworkers, fname=info_file)
    gpu_manager_info = gpu_manager.get_info()
    return gpu_manager, gpu_manager_info

class GPUManager:
    def __init__(self, model_path, *, max_batch, nworkers, fname):
        assert torch.cuda.is_available(), "GPUManager: Error: no cuda device found"
        self.max_batch = max_batch
        self.nworkers = nworkers
        self.memlen = max_batch * nworkers
        mp.set_start_method('spawn', force=True)
        self.request_queue = multiprocessing.Queue()
        self.results_queues = [multiprocessing.Queue() for i in range(self.nworkers)]
        self.events = [multiprocessing.Event() for _ in range(self.nworkers)]
        self.fname = fname
        game = make(2)
        obs, rew, done, info = game.reset()
        state = getstate(obs, info, gettorch=True, unsqueeze=False)
        self.shared_obs = [torch.zeros((self.memlen, *state[i].shape)).share_memory_() for i in range(len(state))]
        self.shared_p = torch.zeros((self.memlen, 52)).share_memory_()
        self.shared_v = torch.zeros((self.memlen, 1)).share_memory_()
        self.model_path = model_path
        self.stop_gpu_manager_event = multiprocessing.Event()
        self.kwargs = {
                "request_queue": self.request_queue,
                "shared_obs": self.shared_obs,
                "shared_p": self.shared_p,
                "shared_v": self.shared_v,
                "events": self.events,
                "model_path": self.model_path,
                "nworker": nworkers,
                "n_batch_per_worker": max_batch,
                "fname": self.fname,
        }
        self.gpu_proc = None

    def get_info(self):
        info_dict = {
                "request_queue": self.request_queue,
                "shared_obs": self.shared_obs,
                "shared_p": self.shared_p,
                "shared_v": self.shared_v,
                "events": self.events,
                "max_batch": self.max_batch,
        }
        return info_dict

    def start(self):
        self.gpu_proc = mp.Process(target = gpu_manager_loop, kwargs=self.kwargs)
        self.gpu_proc.start()

    def stop(self):
        self.request_queue.put((None, None))
        self.gpu_proc.join(timeout=1)
        if self.gpu_proc.is_alive():
            print("Terminating GPU manager loop by force. This may cause corruption!")
            self.gpu_proc.terminate()
            self.gpu_proc.join()


def gpu_manager_loop(*, request_queue, shared_obs, shared_p, shared_v, events, model_path, nworker, n_batch_per_worker, fname):
    with open(fname, "a") as f:
        with redirect_stdout(f):
            os.sched_setaffinity(0, {0,2})
            print("Started GPU Manager")
            print(f"GPU Manager is bound to cores {os.sched_getaffinity(0)}")
            model = PolicyNN()
            model.load_state_dict(torch.load(model_path, weights_only = True))
            model = model.to('cuda').eval()
            cycles = 0
            computed_batches = 0
            now = time()
            batch_size = nworker * n_batch_per_worker
            min_size = batch_size // 4
            max_wait = 0.0005
            while True:
                if time() - now > 10:
                    print(f"GPU Manager Info:")
                    print(f"    Cycles: {cycles}")
                    print(f"    Batches: {computed_batches}")
                    ratio = computed_batches / cycles
                    print(f"    Batches per cycle: {ratio:.1f} (max batch size {batch_size})")
                    print(f"    Batches per second: {round(computed_batches/10)}")
                    cycles = 0
                    computed_batches = 0
                    now = time()

                cycles += 1
                worker_id, n = request_queue.get()
                if worker_id is None:
                    # This is a magic value that the GPU manager sends to signal cease of operation
                    break
                ids = [worker_id]
                ns = [n]
                cumns = [0, n]
                starttime = time()
                while cumns[-1] < min_size and (time() - starttime) < max_wait:
                    try:
                        _id, n = request_queue.get_nowait()
                        ids.append(_id)
                        ns.append(n)
                        cumns.append(cumns[-1] + n)
                    except:
                        continue
                stacked_batches = []
                for i in range(len(ids)):
                    startidx = ids[i] * n_batch_per_worker
                    endidx = startidx + ns[i]
                    stacked_batches.append([shared_obs[i][startidx:endidx].to('cuda', non_blocking=True) for i in range(len(shared_obs))])
                stacked_batches = [torch.cat(tensors) for tensors in zip(*stacked_batches)]
                with torch.no_grad():
                    p_logits, v_head = model(*stacked_batches)
                    p_head = torch.softmax(p_logits, dim=1)
                p_head = p_head.cpu()
                v_head = v_head.cpu()

                for i in range(len(ids)):
                    wid = ids[i]
                    start = wid*n_batch_per_worker
                    end = start + ns[i]
                    start_batch = cumns[i]
                    end_batch = cumns[i+1]
                    shared_p[start:end] = p_head[start_batch:end_batch]
                    shared_v[start:end] = v_head[start_batch:end_batch]

                for wid in ids:
                    events[wid].set()
                computed_batches += sum(ns)
                




