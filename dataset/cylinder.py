import h5py
import torch
from torch.utils.data import Dataset
from einops import repeat

class CylinderMeshDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 time_horizon: int = 25,
                 padding: bool = False,
                 max_len: int = 2060,
                 use_embed: bool = False,
                 ablate: bool=False,
                 return_cells: bool=False) -> None:

        super().__init__()
        f = h5py.File(data_dir, 'r')
        self.n_samples = len(f)

        self.data_dict = {}

        for i in range(self.n_samples):
            self.data_dict[i] = {}
            self.data_dict[i]['u'] = torch.tensor(f[str(i)]['u'][:], dtype=torch.float32)
            self.data_dict[i]['v'] = torch.tensor(f[str(i)]['v'][:], dtype=torch.float32)
            self.data_dict[i]['p'] = torch.tensor(f[str(i)]['pressure'][:], dtype=torch.float32)
            self.data_dict[i]['cells'] = torch.tensor(f[str(i)]['cells'][:], dtype=torch.int32)

            # adjust position and node type to be between 0 and 1
            pos_x = torch.tensor(f[str(i)]['mesh_pos'][:, 0], dtype=torch.float32) # M 2
            pos_y = torch.tensor(f[str(i)]['mesh_pos'][:, 1], dtype=torch.float32) # M 2

            pos_x = (pos_x - pos_x.min()) / (pos_x.max() - pos_x.min())
            pos_y = (pos_y - pos_y.min()) / (pos_y.max() - pos_y.min())

            self.data_dict[i]['mesh_pos'] = torch.stack([pos_x, pos_y], dim=-1)

            node_type = torch.tensor(f[str(i)]['node_type'][:], dtype=torch.float32) # M 1
            node_type = node_type - node_type.min() / (node_type.max() - node_type.min())

            self.data_dict[i]['node_type'] = node_type

            if use_embed and not ablate:
                self.data_dict[i]['prompt'] = f[str(i)]['metadata']['prompt'].asstr()[()]
            
            if ablate:
                # normalize positions (they are on a domain size of [1.6, 0.4])
                cylinder_pos_x = torch.tensor(f[str(i)]['metadata']['center'][0], dtype=torch.float32).unsqueeze(0) / 1.6
                cylinder_pos_y = torch.tensor(f[str(i)]['metadata']['center'][1], dtype=torch.float32).unsqueeze(0) / 0.4 
                cylinder_radius = torch.tensor(f[str(i)]['metadata']['radius'][()], dtype=torch.float32).unsqueeze(0) / 0.4
                inlet_velocity = torch.tensor(f[str(i)]['metadata']['u_inlet'][()], dtype=torch.float32).unsqueeze(0)
                reynolds_number = torch.tensor(f[str(i)]['metadata']['reynolds_number'][()], dtype=torch.float32).unsqueeze(0) / 1000

                if use_embed:
                    cylinder_pos_x = str(cylinder_pos_x.item())
                    cylinder_pos_y = str(cylinder_pos_y.item())
                    cylinder_radius = str(cylinder_radius.item())
                    inlet_velocity = str(inlet_velocity.item())
                    reynolds_number = str(reynolds_number.item())
                    metadata_string = f"{cylinder_pos_x} {cylinder_pos_y} {cylinder_radius} {inlet_velocity} {reynolds_number}"
                    self.data_dict[i]['prompt'] = metadata_string
                else:
                    self.data_dict[i]['prompt_vector'] = torch.cat([cylinder_pos_x, cylinder_pos_y, cylinder_radius, inlet_velocity, reynolds_number], dim=0)
    
        f.close()

        self.time_horizon = time_horizon
        self.time_downs = 100 // self.time_horizon

        self.time = torch.linspace(0, 1, self.time_horizon)

        self.padding = padding 
        self.max_len = max_len 
        self.use_embed = use_embed
        self.ablate = ablate
        self.return_cells = return_cells

        print("Data loaded from: {}".format(data_dir))
        print("n_samples: {}".format(self.n_samples))
        print("Resolution: {}".format(self.data_dict[0]['u'].shape))
        print("Time horizon: {}".format(self.time_horizon))
        print("Using padding: {}".format(self.padding))
        print("Using embeddings: {}".format(use_embed))
        print("Ablating prompt: {}".format(ablate))
        print("Returning cells: {}".format(return_cells))
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def get_return_dict(self, x, pos, cells=None, pad_mask=None, embeddings=None, embed_mask=None, prompt=None, prompt_vector=None):
        return_dict = {'x': x, 'pos': pos}
        if cells is not None:
            return_dict['cells'] = cells
        if pad_mask is not None:
            return_dict['pad_mask'] = pad_mask
        if embeddings is not None:
            return_dict['embeddings'] = embeddings
        if embed_mask is not None:
            return_dict['embed_mask'] = embed_mask
        if prompt is not None:
            return_dict['prompt'] = prompt
        if prompt_vector is not None:
            return_dict['prompt_vector'] = prompt_vector
        return return_dict

    def __getitem__(self, i: int, eval=False):
        t = len(self.time)

        u_i = self.data_dict[i]['u'].unsqueeze(2) # nt, M, 1
        v_i = self.data_dict[i]['v'].unsqueeze(2) # nt, M, 1
        p_i = self.data_dict[i]['p'] # nt, M, 1

        coords_i = repeat(self.data_dict[i]['mesh_pos'], "m n -> t m n", t=t) # nt M 2
        time_i = repeat(self.time, "t -> t m 1", m=u_i.shape[1]) # nt M 1

        u_i = u_i[::self.time_downs]
        v_i = v_i[::self.time_downs]
        p_i = p_i[::self.time_downs]

        cells = None
        pad_mask = None 
        embed_mask = None
        embeddings = None 
        prompt = None
        prompt_vector = None

        if self.padding: # need to pad irregular data to max_len
            length = u_i.shape[1]
            pad_length = self.max_len - length
            u_i = torch.cat([u_i, torch.zeros(t, pad_length, 1)], dim=1) # set to zero so no grads 
            v_i = torch.cat([v_i, torch.zeros(t, pad_length, 1)], dim=1)
            p_i = torch.cat([p_i, torch.zeros(t, pad_length, 1)], dim=1)

            # set positions and time to -1 (not physically possible so will be ignored by GINO)
            coords_i = torch.cat([coords_i, -1 * torch.ones(t, pad_length, 2)], dim=1) 
            time_i = torch.cat([time_i, -1 * torch.ones(t, pad_length, 1)], dim=1)

            pad_mask = torch.zeros(self.max_len) # M
            pad_mask[:length] = 1 # set to 1 where we have data

        x = torch.cat([u_i, v_i, p_i], dim=-1) # nt M 3
        pos = torch.cat([coords_i, time_i], dim=-1) # nt M 3

        if eval or self.return_cells:
            cells = self.data_dict[i]['cells']

        if self.ablate and not self.use_embed:
            prompt_vector = self.data_dict[i]['prompt_vector']

        if self.use_embed:
            prompt = self.data_dict[i]['prompt']

        return_dict = self.get_return_dict(x, pos, cells, pad_mask, embeddings, embed_mask, prompt, prompt_vector)
        return return_dict
            

