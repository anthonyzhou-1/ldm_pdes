from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os.path
import torch 
import numpy as np
from tqdm import tqdm

class Normalizer:
    def __init__(self,
                 use_norm = False,
                 stat_path = "./",
                 dataset=None,
                 scaler = "normal",
                 recalculate = False,
                 conditional = True):
        self.use_norm = use_norm
        self.conditional = conditional
        self.scaler = scaler # normal or minmax
       
        if not self.use_norm:
            print("Normalization is turned off")
            return 

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
            print("Statistics loaded from", stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            print("Calculating statistics for normalization")
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
            for batch in tqdm(dataloader):
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
                if self.conditional:
                    pickle.dump([self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale, self.cond_min, self.cond_scale], f)
                else:
                    pickle.dump([self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale], f)
        else:
            with open(path, "wb") as f:
                if self.conditional:
                    pickle.dump([self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std, self.cond_min, self.cond_scale], f)
                else:
                    pickle.dump([self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            if self.scaler == "minmax":
                if self.conditional:
                    self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale, self.cond_min, self.cond_scale = pickle.load(f)
                else:
                    self.u_min, self.u_scale, self.v_min, self.v_scale, self.p_min, self.p_scale = pickle.load(f)
                    self.cond_min = None 
                    self.cond_scale = None
            else:
                if self.conditional:
                    self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std, self.cond_min, self.cond_scale = pickle.load(f)
                else:
                    self.u_mean, self.u_std, self.v_mean, self.v_std, self.p_mean, self.p_std = pickle.load(f)
                    self.cond_min = None 
                    self.cond_scale = None

    def normalize_cond(self, cond):
        cond_norm = cond.clone()
        cond_norm = cond_norm * self.cond_scale + self.cond_min

        return cond_norm
    
    def denormalize_cond(self, cond):
        cond_denorm = cond.clone()
        cond_denorm = (cond_denorm - self.cond_min) / self.cond_scale

        return cond_denorm
    
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