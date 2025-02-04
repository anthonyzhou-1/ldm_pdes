import h5py
import torch
from torch.utils.data import Dataset
import numpy as np 
from dataset.turb_labels import TurbLabels

class Turb3DDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 split: str,
                 time_horizon: int = 500,
                 time_window: int = 240,
                 time_stride: int = 5,
                 use_embed: bool = False,
                 first_time_only: bool = False) -> None:

        super().__init__()
        self.split = split
        # manually list all possible data directories
        if self.split == "train":
            self.fnames = ['step-higher', 'step-lower', 'corner', 'opp-corners-asym', 'neighbor-corners', 'corners', 'pillar', 'offset-pillar', 'double-pillar', 'opp-pillar', 'bar', 'double-bar', 'teeth', 'offset-teeth', 'elbow', 'wide-elbow', 'elbow-snug', 'open-elbow', 'donut', 'U', 'H', 'T', 'disjoint-T', 'plus', 'minus', 'square', 'square-offset', '2x2', '2x2-large', '3x3', '3x3-inv', 'cross-wide', 'cross-offset', 'platform', 'high-platform', 'altar' ]
        elif self.split == "valid":
            self.fnames = ['step-low', 'opp-corners-sym', 'wide-pillar', 'elbow-asym', 'square-large', 'cross', 'wide-teeth', 'step-high', 'offset-bar']

        self.n_samples = len(self.fnames) # set the length of the dataset to number of geometries. In reality, each geometry has multiple data samples.
        self.first_time_only = first_time_only
        self.file_pointers = []

        for i in range(self.n_samples):
            path = f"{data_dir}/{self.fnames[i]}/data.h5"
            pointer = h5py.File(path, 'r')
            self.file_pointers.append(pointer)

        self.time_horizon = time_horizon
        self.time_window = time_window
        self.time_stride = time_stride

        self.time = torch.linspace(0, 1, self.time_horizon)
        self.use_embed = use_embed

        if self.use_embed:
            self.labeler = TurbLabels(split=split)
            self.labels = []
            for i in range(self.n_samples):
                self.labels.append(self.labeler.get_label(i))
            
            print("Initialized Turb3DDataset with labels")
            
        print("Data loaded from: {}".format(data_dir))
        print("n_samples: {}".format(self.n_samples))
        print("Time horizon: {}".format(self.time_horizon))
        print("Time window: {}".format(self.time_window))
        print("Time stride: {}".format(self.time_stride))
        if first_time_only:
            print("Only using first timesteps from dataset")
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def get_return_dict(self, x, mask=None, prompt=None):
        return_dict = {'x': x}
        if mask is not None:
            return_dict['mask'] = mask
        if prompt is not None:
            return_dict['prompt'] = prompt
        return return_dict
    
    def load_data(self, f, idx, features = ["u", "p"]):
        """Load data from a data.h5 file into an easily digestible matrix format.

        Arguments
        ---------
        f
            pointer to a data.h5 file in the `shapes` dataset
        idx
            Index or indices of sample to load. Can be a number, list, boolean mask or a slice.
        features
            Features to load. By default loads only velocity and pressure but you can also
            access the LES specific k and nut variables.

        Returns
        -------
        t: np.ndarray of shape T
            Time steps of the loaded data frames
        data_3d: np.ndarray of shape T x W x H x D x F
            3D data with all features concatenated in the order that they are requested, i.e.
            in the default case the first 3 features will be the velocity vector and the fourth
            will be the pressure
        inside_mask: np.ndarray of shape W x H x D
            Boolean mask that marks the inside cells of the domain, i.e. cells that are not part
            of walls, inlets or outlets
        """

        t = np.array(f["data/times"])

        cell_data = np.concatenate([np.atleast_3d(f["data"][name][idx]) for name in features], axis=-1)
        padded_cell_counts = np.array(f["grid/cell_counts"])
        cell_idx = np.array(f["grid/cell_idx"])

        n_steps, n_features = cell_data.shape[0], cell_data.shape[-1]
        data_3d = np.zeros((n_steps, *padded_cell_counts, n_features))
        data_3d.reshape((n_steps, -1, n_features))[:, cell_idx] = cell_data

        inside_mask = np.zeros(padded_cell_counts, dtype=bool)
        inside_mask.reshape(-1)[cell_idx] = 1

        return t, data_3d, inside_mask
    
    def downsample(self, x):
        x = x[:, 2:, 1:49, 1:49, :] # 48, 192, 48, 48, 4
        x = x[:, ::2, ::2, ::2, :] # 48, 96, 24, 24, 4
        return x 
    
    def downsample_mask(self, mask):
        mask = mask[2:, 1:49, 1:49] # 192, 48, 48
        mask = mask[::2, ::2, ::2] # 96, 24, 24

        return mask

    def __getitem__(self, i: int, eval=False):
        f = self.file_pointers[i] # get the file pointer for the i-th geometry

        if eval or self.first_time_only: 
            start_idx = 0
        else:
            start_idx = np.random.randint(0, self.time_horizon - self.time_window)

        time_idx = np.arange(start_idx, start_idx + self.time_window, self.time_stride)

        # raw data should be in shape (t x w x h x d x f) (48 x 196 x 50 x 50 x 4)
        t, x, mask = self.load_data(f, time_idx) # gets raw data from the file pointer. 
        x = x.astype(np.float32)
        x = self.downsample(x)
        mask = self.downsample_mask(mask)

        prompt = None 
        if self.use_embed:
            prompt = self.labels[i]

        return_dict = self.get_return_dict(x, mask, prompt)
        return return_dict
            

