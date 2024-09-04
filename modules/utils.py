import yaml
import torch
import importlib
import numpy as np
from collections import abc
import multiprocessing as mp
from threading import Thread
from queue import Queue
from inspect import isfunction

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res
    


from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os.path
import torch 
import numpy as np

class Normalizer:
    def __init__(self,
                 use_norm = False,
                 stat_path = "./",
                 dataset=None,
                 scaler = "normal",
                 recalculate = False):
        self.use_norm = use_norm
        self.scaler = scaler # normal or minmax
       
        if not self.use_norm:
            print("Normalization is turned off")
            return 

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            
            dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=0)
            
            if self.scaler == "normal":
                u_scaler = StandardScaler()
                v_scaler = StandardScaler()
                p_scaler = StandardScaler()
            elif self.scaler == "minmax":
                u_scaler = MinMaxScaler(feature_range=(-1, 1)) # -1 to 1 for ldm 
                v_scaler = MinMaxScaler(feature_range=(-1, 1))
                p_scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError("Scaler must be either 'normal' or 'minmax'")
            
            cond_scaler = MinMaxScaler(feature_range=(0, 1)) # make cond scaler always minmax
            self.cond_min = None
            self.cond_scale = None

            max_len = 0
            for batch in dataloader:
                data = batch["x"]

                pad_mask = batch.get("pad_mask", None)
                cond = batch.get("cond", None)

                if pad_mask is not None:
                    data_length = torch.sum(pad_mask, dtype=torch.long)
                    data = data[:, :, :data_length]

                u, v, p = self.get_data(data)

                length = u.shape[2]

                if length > max_len:
                    max_len = length

                u_scaler.partial_fit(u.reshape(-1, 1))
                v_scaler.partial_fit(v.reshape(-1, 1))
                p_scaler.partial_fit(p.reshape(-1, 1))

                if cond is not None:
                    cond_scaler.partial_fit(cond.reshape(-1, 1))

            if self.scaler == "normal":
                self.u_mean = u_scaler.mean_.item()
                self.u_std = np.sqrt(u_scaler.var_).item()
                self.v_mean = v_scaler.mean_.item()
                self.v_std = np.sqrt(v_scaler.var_).item()
                self.p_mean = p_scaler.mean_.item()
                self.p_std = np.sqrt(p_scaler.var_).item()

            else:
                self.u_min = u_scaler.min_.item()
                self.u_scale = u_scaler.scale_.item()
                self.v_min = v_scaler.min_.item()
                self.v_scale = v_scaler.scale_.item()
                self.p_min = p_scaler.min_.item()
                self.p_scale = p_scaler.scale_.item()

            if cond is not None:
                self.cond_min = cond_scaler.min_.item()
                self.cond_scale = cond_scaler.scale_.item()
                
            self.save_stats(path=stat_path)
            print("Statistics saved to", stat_path)
            print("max_len:", max_len)

        self.print_stats()

        if self.scaler == "normal":
            self.u_mean = torch.tensor(self.u_mean)
            self.u_std = torch.tensor(self.u_std)
            self.v_mean = torch.tensor(self.v_mean)
            self.v_std = torch.tensor(self.v_std)
            self.p_mean = torch.tensor(self.p_mean)
            self.p_std = torch.tensor(self.p_std)

        else:
            self.u_min = torch.tensor(self.u_min)
            self.u_scale = torch.tensor(self.u_scale)
            self.v_min = torch.tensor(self.v_min)
            self.v_scale = torch.tensor(self.v_scale)
            self.p_min = torch.tensor(self.p_min)
            self.p_scale = torch.tensor(self.p_scale)


    def get_data(self, data):
        # data in shape [batch, t, nx, ny, c] or [batch, t, n, c]
        u = data[..., 0]
        v = data[..., 1]
        p = data[..., 2]

        return u, v, p
    
    def assemble_data(self, u, v, p):
        return torch.stack([u, v, p], dim=-1)

    def print_stats(self):
        if self.scaler == "minmax":
            print(f"u min: {self.u_min}, u scale: {self.u_scale}")
            print(f"v min: {self.v_min}, v scale: {self.v_scale}")
            print(f"p min: {self.p_min}, p scale: {self.p_scale}")
        else:
            print(f"u mean: {self.u_mean}, u std: {self.u_std}")
            print(f"v mean: {self.v_mean}, v std: {self.v_std}")
            print(f"p mean: {self.p_mean}, p std: {self.p_std}")

        if self.cond_min is not None:
            print(f"cond min: {self.cond_min}, cond scale: {self.cond_scale}")

    def save_stats(self, path):
        if self.scaler == "minmax":
            with open(path, "wb") as f:
                pickle.dump([self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale, self.cond_min, self.cond_scale], f)
        else:
            with open(path, "wb") as f:
                pickle.dump([self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std, self.cond_min, self.cond_scale], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            if self.scaler == "minmax":
                self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale, self.cond_min, self.cond_scale = pickle.load(f)
            else:
                self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std, self.cond_min, self.cond_scale = pickle.load(f)

    def normalize(self, x, cond = None):
        if not self.use_norm:
            return x

        x_norm = x.clone()
        u_norm, v_norm, p_norm = self.get_data(x_norm)

        if self.scaler == "normal":
            u_norm = (u_norm - self.u_mean) / self.u_std
            v_norm = (v_norm - self.v_mean) / self.v_std
            p_norm = (p_norm - self.p_mean) / self.p_std

        else:
            u_norm = u_norm * self.u_scale + self.u_min
            v_norm = v_norm * self.v_scale + self.v_min
            p_norm = p_norm * self.p_scale + self.p_min

        if cond is not None:
            cond_norm = cond.clone()
            cond_norm = cond_norm * self.cond_scale + self.cond_min

        x_norm = self.assemble_data(u_norm, v_norm, p_norm)
        
        if cond is not None:
            return x_norm, cond_norm

        return x_norm

    def denormalize(self, x, cond = None): 
        if not self.use_norm:
            return x

        x_denorm = x.clone()
        u_denorm, v_denorm, p_denorm = self.get_data(x_denorm)

        if self.scaler == "normal":
            u_denorm = u_denorm * self.u_std + self.u_mean
            v_denorm = v_denorm * self.v_std + self.v_mean
            p_denorm = p_denorm * self.p_std + self.p_mean

        else:
            u_denorm = (u_denorm - self.u_min) / self.u_scale
            v_denorm = (v_denorm - self.v_min) / self.v_scale
            p_denorm = (p_denorm - self.p_min) / self.p_scale

        if cond is not None:
            cond_denorm = cond.clone()
            cond_denorm = (cond_denorm - self.cond_min) / self.cond_scale

        x_denorm = self.assemble_data(u_denorm, v_denorm, p_denorm)

        if cond is not None:
            return x_denorm, cond_denorm

        return x_denorm