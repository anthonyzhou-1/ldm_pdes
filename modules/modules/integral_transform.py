import torch
from torch import nn
import torch.nn.functional as F

from typing import Literal
import importlib
from torch import einsum

#https://github.com/neuraloperator/neuraloperator

def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: Literal["mean", "sum"],
    use_scatter=True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or torch_scatter is not
    properly built, segment_csr falls back to a naive PyTorch implementation

    Note: the native version is mainly intended for running tests on 
    CPU-only GitHub CI runners to get around a versioning issue. 
    torch_scatter should be installed and built if possible. 

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if (
        importlib.util.find_spec("torch_scatter") is not None
        and use_scatter
    ):
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduce)

    else:
        if use_scatter:
            print("Warning: use_scatter is True but torch_scatter is not properly built. \
                  Defaulting to naive PyTorch implementation")
        # if batched, shape [b, n_reps, channels]
        # otherwise shape [n_reps, channels]
        if src.ndim == 3:
            batched = True
            point_dim = 1
        else:
            batched = False
            point_dim = 0

        # if batched, shape [b, n_out, channels]
        # otherwise shape [n_out, channels]
        output_shape = list(src.shape)
        n_out = indptr.shape[point_dim] - 1
        output_shape[point_dim] = n_out

        out = torch.zeros(output_shape, device=src.device)

        for i in range(n_out):
            # reduce all indices pointed to in indptr from src into out
            if batched:
                from_idx = (slice(None), slice(indptr[0,i], indptr[0,i+1]))
                ein_str = 'bio->bo'
                start = indptr[0,i]
                n_nbrs = indptr[0,i+1] - start
                to_idx = (slice(None), i)
            else:
                from_idx = slice(indptr[i], indptr[i+1])
                ein_str = 'io->o'
                start = indptr[i]
                n_nbrs = indptr[i+1] - start
                to_idx = i
            src_from = src[from_idx]
            if n_nbrs > 0:
                to_reduce = einsum(ein_str, src_from)
                if reduce == "mean":
                    to_reduce /= n_nbrs
                out[to_idx] += to_reduce
        return out
    
# Reimplementation of the MLP class using Linear instead of Conv
class MLPLinear(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
    
class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined
    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on
           each x) over which to integrate
    k : A kernel parametrized as a MLP
    f : Input function to integrate against given
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the MLPLinear class
    mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        MLPLinear class. Only used if mlp_layers is
        given and mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        Whether to use torch_scatter's implementation of 
        segment_csr or our native PyTorch version. torch_scatter 
        should be installed by default, but there are known versioning
        issues on some linux builds of CPU-only PyTorch. Try setting
        to False if you experience an error from torch_scatter.
    """

    def __init__(
        self,
        mlp=None,
        mlp_layers=None,
        mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_torch_scatter=True,
    ):
        super().__init__()

        assert mlp is not None or mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter

        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if mlp is None:
            self.mlp = MLPLinear(layers=mlp_layers, non_linearity=mlp_non_linearity)
        else:
            self.mlp = mlp
            

    """"
    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]] # Riemann points in physical domain. 
        #Consists of all points in each neighborhood for each latent point x. Shape is [n_riepts, pos_dim]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        # f_y has a batch dim IFF batched=True
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors["neighbors_index"]] # Get input features, which are function values for each riemann point
                # Shape is [n_reipts, f_dim]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        ) # Gets number of riemann points to integrate over for each latent point

        self_features = torch.repeat_interleave(x, num_reps, dim=0) # Repeat each latent point to the number of Riemann points
        # shape is [n_reipts, latent_pos_dim] , latent_pos_dim should equal pos_dim

        agg_features = torch.cat([rep_features, self_features], dim=-1) # Concat latent and Riemann points, shape is [n_riepts, pos_dim + latent_pos_dim]
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features = self.mlp(agg_features) # Apply kernel to concatenated latent and Riemann points. Shape is [n_riepts, f_dim]

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features # Multiply kernel by function values. Shape is [n_reps, f_dim]

        if weights is not None: # calculate weights to sum over Riemann points for each latent point
            assert weights.ndim == 1, "Weights must be of dimension 1 in all cases"
            nbr_weights = weights[neighbors["neighbors_index"]]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat(
                    [batch_size] + [1] * nbr_weights.ndim
                )
            rep_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean"

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        # For each latent point, take the mean of the kernel values over the Riemann points
        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        return out_features