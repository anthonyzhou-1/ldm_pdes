import functools
from typing import Callable, Optional

import h5py
import torch
import torchdata.datapipes as dp

class NavierStokesDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load Navier-Stokes dataset.

    Args:
        dp (dp.iter.IterDataPipe): List of `hdf5` files containing Navier-Stokes data.
        mode (str): Mode to load data from. Can be one of `train`, `val`, `test`.
        limit_trajectories (int, optional): Limit the number of trajectories to load from individual `hdf5` file. Defaults to None.
        usegrid (bool, optional): Whether to output spatial grid or not. Defaults to False.

    Yields:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]): Tuple containing particle scalar field, velocity vector field, and optionally buoyancy force parameter value  and spatial grid.
    """

    def __init__(self, dp, mode: str,
                 limit_trajectories: Optional[int] = None,
                 usegrid: bool = False, conditioned: bool = False, use_embed:bool = False) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        self.conditioned = conditioned
        self.use_embed = use_embed
        self.replace_newlines = True 

    def __iter__(self):
        for path in self.dp:
            # print('rank: ', os.environ['RANK'])
            # print('seed of path: ', path[-11:])
            try:
                with h5py.File(path, "r") as f:
                    data = f[self.mode]
                    if self.limit_trajectories is None or self.limit_trajectories == -1:
                        num = data["u"].shape[0]
                    else:
                        num = self.limit_trajectories

                    iter_start = 0
                    iter_end = num

                    for idx in range(iter_start, iter_end):
                        u = torch.tensor(data["u"][idx]) # t nx ny
                        vx = torch.tensor(data["vx"][idx]) # t nx ny
                        vy = torch.tensor(data["vy"][idx]) # t nx ny
                        if "buo_y" in data and self.conditioned:
                            cond = torch.tensor(data["buo_y"][idx]).unsqueeze(0).float()
                        else:
                            cond = None

                        if self.use_embed:
                            all_text = data['labels']
                            text = all_text[str(idx)].asstr()[()]
                            #print(text)
                            if self.replace_newlines: # replace newlines with spaces
                                text = text.replace('\n', ' ')

                        x = torch.cat((vx.unsqueeze(-1), vy.unsqueeze(-1), u.unsqueeze(-1)), dim=-1) # t nx ny 3
                        x = x.float()

                        if self.usegrid:
                            gridx = torch.linspace(0, 1, data["x"][idx].shape[0])
                            gridy = torch.linspace(0, 1, data["y"][idx].shape[0])
                            gridx = gridx.reshape(
                                1,
                                gridx.size(0),
                                1,
                            ).repeat(
                                1,
                                1,
                                gridy.size(0),
                            )
                            gridy = gridy.reshape(
                                1,
                                1,
                                gridy.size(0),
                            ).repeat(
                                1,
                                gridx.size(1),
                                1,
                            )
                            grid = torch.cat((gridx[:, None], gridy[:, None]), dim=1)
                        else:
                            grid = None

                        if self.use_embed:
                            yield x, cond, grid, text
                        else:
                            yield x, cond, grid
            except OSError as e:
                print(f"Error opening file {path}: {e}")
                continue


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname

class ConditionedSmokeData(dp.iter.IterDataPipe):
    """Data for evaluation of time conditioned PDEs

    Args:
        dp (torchdata.datapipes.iter.IterDataPipe): Data pipe that returns individual PDE trajectories.
        trajlen (int): Length of a trajectory in the dataset.
        delta_t (int): Evaluates predictions conditioned at that delta_t.

    Tip:
        Make sure `delta_t` is less than half of `trajlen`.
    """

    def __init__(self, dp: dp.iter.IterDataPipe,
                 trajlen: int,
                 start_time: int,   # start time for prediction
                 delta_t: int,
                 downsample: int = 1,
                 use_embed: bool = False) -> None:
        super().__init__()
        self.dp = dp
        self.trajlen = trajlen
        if 2 * delta_t >= self.trajlen:
            raise ValueError("delta_t should be less than half the trajectory length")

        self.delta_t = delta_t
        self.start_time = start_time
        self.downsample = downsample
        self.use_embed = use_embed

    def __iter__(self):
        begin = self.start_time
        if self.trajlen - begin < self.delta_t:
            raise ValueError("Trajectory length is less than delta_t + start_time")
        if (self.trajlen - begin) % self.delta_t != 0:
            raise Warning("Trajectory length is not divisible by delta_t. Some time steps will be ignored.")

        if self.use_embed:
            for x, cond, grid, text in self.dp:
                x = x[:, ::self.downsample, ::self.downsample, :] # t nx//d ny//d 3
                x = x[:self.trajlen] # truncate to trajlen
                x = x[begin::self.delta_t, ...] # trajlen//dt nx//d ny//d 3

                out_dict = {"x": x}

                if cond is not None:
                    out_dict["cond"] = cond
                if grid is not None:
                    out_dict["grid"] = grid
                if text is not None:
                    out_dict["prompt"] = text

                yield out_dict

        else:
            for x, cond, grid in self.dp:
                x = x[:, ::self.downsample, ::self.downsample, :] # t nx//d ny//d 3
                x = x[:self.trajlen] # truncate to trajlen
                x = x[begin::self.delta_t, ...] # trajlen//dt nx//d ny//d 3

                out_dict = {"x": x}

                if cond is not None:
                    out_dict["cond"] = cond
                if grid is not None:
                    out_dict["grid"] = grid

                yield out_dict


def build_datapipes(
    data_config,
    usegrid: bool,
    dataset_opener: Callable[..., dp.iter.IterDataPipe],
    lister: Callable[..., dp.iter.IterDataPipe],
    sharder: Callable[..., dp.iter.IterDataPipe],
    filter_fn: Callable[..., dp.iter.IterDataPipe],
    mode: str,
):
    """Build datapipes for training and evaluation.

    Args:
        data_path (str): Path to the data.
        limit_trajectories (int): Number of trajectories to use.
        usegrid (bool): Whether to use spatial grid as input.
        dataset_opener (Callable[..., dp.iter.IterDataPipe]): Dataset opener.
        lister (Callable[..., dp.iter.IterDataPipe]): List files.
        sharder (Callable[..., dp.iter.IterDataPipe]): Shard files.
        filter_fn (Callable[..., dp.iter.IterDataPipe]): Filter files.
        mode (str): Mode of the data. ["train", "valid", "test"]
        time_history (int, optional): Number of time steps in the past. Defaults to 1.
        time_future (int, optional): Number of time steps in the future. Defaults to 1.
        time_gap (int, optional): Number of time steps between the past and the future to be skipped. Defaults to 0.
        onestep (bool, optional): Whether to use one-step prediction. Defaults to False.
        conditioned (bool, optional): Whether to use conditioned data. Defaults to False.
        delta_t (Optional[int], optional): Time step size. Defaults to None. Only used for conditioned data.
        conditioned_reweigh (bool, optional): Whether to reweight conditioned data. Defaults to True.

    Returns:
        dpipe (IterDataPipe): IterDataPipe for training and evaluation.
    """
    print("Building datapipe")
    dpipe = lister(
        data_config.data_path,
    )
    dpipe = dpipe.filter(filter_fn)
    # enforce even number of files 
    num_files = len(list(dpipe))
    if num_files % 2 != 0:
        dpipe = dpipe.header(2 * (num_files) // 2)
    print(mode)
    print("Number of files: ", num_files)

    if mode == "train":
        dpipe = dpipe.shuffle(buffer_size=32*num_files) 

    if mode == "train":

        dpipe = dataset_opener(
            sharder(dpipe),
            mode=mode,
            limit_trajectories=data_config.num_training_trajs,
            usegrid=usegrid,
            use_embed = data_config.use_embed,
        )

        dpipe = ConditionedSmokeData(
            dpipe,
            data_config.trajlen,
            data_config.start_time,
            data_config.delta_t,
            data_config.downsample,
            data_config.use_embed,
        )

    else:
        dpipe = dataset_opener(
            sharder(dpipe),
            mode=mode,
            limit_trajectories=data_config.num_testing_trajs,
            usegrid=usegrid,
            use_embed = data_config.use_embed,
        )

        dpipe = ConditionedSmokeData(
            dpipe,
            data_config.trajlen,
            data_config.start_time,
            data_config.delta_t,
            data_config.downsample,
            data_config.use_embed,
        )

    return dpipe


train_datapipe_ns_cond = functools.partial(
    build_datapipes,
    usegrid=False,
    dataset_opener=functools.partial(NavierStokesDatasetOpener, conditioned=True, usegrid=False),
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)

valid_datapipe_ns_cond = functools.partial(
    build_datapipes,
    usegrid=False,
    dataset_opener=functools.partial(NavierStokesDatasetOpener, conditioned=True, usegrid=False),
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
)

test_datapipe_ns_cond = functools.partial(
    build_datapipes,
    usegrid=False,
    dataset_opener=functools.partial(NavierStokesDatasetOpener, conditioned=True, usegrid=False),
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
)
