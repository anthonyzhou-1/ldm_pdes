import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from modules.models.ddpm import mean_flat
from scipy.special import roots_legendre
import pickle 
from pathlib import Path

#https://github.com/martenlienen/generative-turbulence
#https://github.com/pdearena/pdearena

def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


def pearson_correlation(input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = False):
    B = input.size(0)
    T = input.size(1)
    input = input.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    input_mean = torch.mean(input, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    input_std = torch.std(input, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((input - input_mean) * (target - target_mean), dim=2) / (input_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr.squeeze() 


class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)


class PearsonCorrelationScore(torch.nn.Module):
    """Pearson Correlation Score for PDEs."""

    def __init__(self, channel: int = None, reduce_batch: bool = False) -> None:
        super().__init__()
        self.channel = channel
        self.reduce_batch = reduce_batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            input = input[:, :, self.channel]
            target = target[:, :, self.channel]
        # input in shape b nx nt or b nx ny nt
        if len(input.shape) == 3:
            input = rearrange(input, "b nx nt -> b nt nx")
            target = rearrange(target, "b nx nt -> b nt nx")
        elif len(input.shape) == 4:
            if input.shape[-1] != 1:
                input = rearrange(input, "b nx ny c -> b 1 nx ny c") # add time dimension
                target = rearrange(target, "b nx ny c -> b 1 nx ny c")
            else:
                input = rearrange(input, "b nx ny nt -> b nt nx ny") # assume time dimension is 1
                target = rearrange(target, "b nx ny nt -> b nt nx ny")
        
        return pearson_correlation(input, target, reduce_batch=self.reduce_batch)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    # adopt weight if global step is below threshold
    # Used to give the generator a head start 
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
                 discriminator, 
                 lpips, 
                 disc_start, 
                 logvar_init=0.0, 
                 kl_weight=1.0, 
                 pixelloss_weight=1.0,
                 disc_factor=1.0, 
                 disc_weight=1.0,
                 perceptual_weight=0.0, 
                 disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = lpips
        self.perceptual_weight = perceptual_weight 
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = discriminator
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        # calculate the adaptive weight for the discriminator to scale discriminator loss to about the norm of the nll loss
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, 
                inputs, 
                reconstructions, 
                posteriors, 
                optimizer_idx,
                global_step, 
                last_layer=None, 
                cond=None, 
                split="train",
                weights=None, 
                normalizer=None, 
                disc_pos = None,
                disc_latent = None,
                mask = None,
                pad_mask = None):

        if split == "val":
            # calculate l1 loss during val on unnormalized data
            rec_loss = torch.abs(normalizer.denormalize(inputs) - normalizer.denormalize(reconstructions)) / torch.norm(inputs, p=1)
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) / torch.norm(inputs, p=1) 

        if mask is not None:
            mask = repeat(mask, 'b nx ny -> b 1 1 nx ny') # mask is zero where data exists, one at obstacles/boundaries
            rec_loss = rec_loss * (1-mask) # propagate only the error where mask is zero
            reconstructions = normalizer.normalize(normalizer.denormalize(reconstructions) * (1-mask))
            # reconstruction needs to have normalized values at masked positions for discriminator
            # otherwise discriminator can tell between rec and input by comparing normalized and unnormalized values

        if pad_mask is not None:
            rec_loss = rec_loss * pad_mask # only propagate loss where mask is 1
            reconstructions = reconstructions * pad_mask # set padded positions to zero for discriminator

        if self.perceptual_weight > 0 and split == "train":
            if mask is not None:
                mask = repeat(mask, 'b 1 1 nx ny -> b 1 nt nx ny', nt=inputs.shape[2])
            else:
                b, _, nt, nx, ny = inputs.shape
                mask = torch.ones(b, 1, nt, nx, ny, device=inputs.device) # add mask for lpips

            inputs_lpips = torch.cat((inputs, mask), dim=1)
            rec_lpips = torch.cat((reconstructions, mask), dim=1)

            p_loss = self.perceptual_loss(inputs_lpips, rec_lpips) # shape [b, 1, nt, 1, 1]; averaged over c, nx, ny
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if disc_pos is not None:
                logits_fake = self.discriminator(reconstructions, disc_pos, disc_latent, pad_mask) # wants to be -1
            elif cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous(), cond)

            # when generator is perfect, g_loss should be around -1, when discriminator is perfect, g_loss should be around 1
            g_loss = -torch.mean(logits_fake) # generator wants to fool discriminator, therefore we maximize logits_fake 

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # weighted_nll_loss ~ 1e4, kl_loss ~ 1e4-1e5, kl_weight ~ 1e-6, g_loss ~ -1e1-1e2, d_weight ~ 1e3
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor, device=inputs.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if disc_pos is not None:
                logits_real = self.discriminator(inputs.contiguous().detach(), disc_pos, disc_latent, pad_mask)  # wants to be 1
                logits_fake = self.discriminator(reconstructions.contiguous().detach(), disc_pos, disc_latent, pad_mask) # wants to be -1
            elif cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(inputs.contiguous().detach(), cond)
                logits_fake = self.discriminator(reconstructions.contiguous().detach(), cond)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
                }
            return d_loss, log
        
class KL_Loss():
    # simple VAE loss without discriminator or perceptual loss
    def __init__(self, 
                 kl_weight=1.0, ):

        super().__init__()
        self.kl_weight = kl_weight

    def __call__(self, 
                inputs, 
                reconstructions, 
                posteriors, 
                split="train",
                normalizer=None, 
                mask = None,
                pad_mask = None):

        if split == "val":
            # calculate l1 loss during val on unnormalized data
            rec_loss = torch.abs(normalizer.denormalize(inputs) - normalizer.denormalize(reconstructions)) 
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # shape [b, c, nt, nx, ny]

        if mask is not None:
            rec_loss = rec_loss * mask # propagate only the error where mask is True/one

        if pad_mask is not None:
            rec_loss = rec_loss * pad_mask # only propagate loss where mask is 1

        rec_loss = torch.sum(mean_flat(rec_loss)) / rec_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss_unweighted = kl_loss.clone().detach().mean()
        kl_loss = self.kl_weight * kl_loss

        # rec_loss ~ 1e-1, kl_loss ~ 1e4-1e5, kl_weight ~ 1e-6
        loss = rec_loss + kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/kl_loss_unweighted".format(split): kl_loss_unweighted,
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log

def interp3(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Tri-linearly interpolate data from regular 3D grids onto arbitrary points.

    Arguments
    ---------
    grid
        F-dimensional features at integer grid coordinates with shape (..., F, X, Y, Z)
    points
        3D points with shape (N, 3)

    Returns
    -------
    Interpolation values with shape (..., N, F)
    """

    # Find the 8 neighboring grid point coordinates of each query point
    p0 = torch.floor(points).long()
    p1 = p0 + 1
    x0, y0, z0 = torch.unbind(torch.floor(points).long(), dim=-1)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Ensure indices are within grid bounds
    lower_bounds = grid.new_zeros(3, dtype=torch.long)
    upper_bounds = grid.new_tensor(grid.shape[-3:], dtype=torch.long) - 1
    p0 = torch.clamp(p0, lower_bounds, upper_bounds)
    p1 = torch.clamp(p1, lower_bounds, upper_bounds)

    x0, y0, z0 = torch.unbind(p0, dim=-1)
    x1, y1, z1 = torch.unbind(p1, dim=-1)

    # Compute the interpolation weights
    wx, wy, wz = torch.unbind(points - p0, dim=-1)

    # Use the weights to compute the interpolated value
    return (
        (1 - wx) * (1 - wy) * (1 - wz) * grid[..., x0, y0, z0]
        + (1 - wx) * (1 - wy) * wz * grid[..., x0, y0, z1]
        + (1 - wx) * wy * (1 - wz) * grid[..., x0, y1, z0]
        + (1 - wx) * wy * wz * grid[..., x0, y1, z1]
        + wx * (1 - wy) * (1 - wz) * grid[..., x1, y0, z0]
        + wx * (1 - wy) * wz * grid[..., x1, y0, z1]
        + wx * wy * (1 - wz) * grid[..., x1, y1, z0]
        + wx * wy * wz * grid[..., x1, y1, z1]
    )


class TurbulentKineticEnergySpectrum(nn.Module):
    """Estimate the turbulent kinetic energy spectrum of a 3D flow field."""

    def __init__(self, n: int = 5810, path=None):
        """
        Arguments
        ---------
        n: Number of grid points for Lebedev quadrature
        """

        super().__init__()
        if path is None:
            numgrids_file = Path("configs/turb3d/numgrids.pickle")
        else:
            numgrids_file = Path(path)
        numgrids = pickle.loads(numgrids_file.read_bytes())

        self.n = n
        if n not in numgrids:
            raise RuntimeError(f"n={n} is not supported by numgrid.")

        x, y, z, w = numgrids[n]
        p = torch.tensor([x, y, z]).T
        w = torch.tensor(w)

        self.register_buffer("p", p.float())
        self.register_buffer("w", w.float())

    def forward(self, u_perturbation: torch.Tensor, k: torch.Tensor):
        # Compute the TKE at each point (Pope, p. 88)
        tke = 0.5 * (u_perturbation**2).sum(dim=-4)

        # Fourier-transform the TKE
        tke_fft = torch.fft.fftn(tke, dim=(-3, -2, -1))
        tke_fft = torch.fft.fftshift(tke_fft, dim=(-3, -2, -1))

        # Construct the query points for interpolation by centering a sphere on the zero
        # frequency in the shifted FFT and scaling it to radius k
        center = k.new_tensor([s // 2 for s in u_perturbation.shape[-3:]])
        p_query = k[:, None, None] * self.p + center

        # Integrate the directional squared frequency amplitudes of the spectrum over a
        # sphere of radius exactly by interpolating the values from the grid points onto
        # the sphere in the log-domain. We interpolate in the log-domain, because the
        # magnitudes decay exponentially with increasing k which is badly approximated
        # by a linear function and leads to overestimation of the energies, i.e. a shift
        # up in a TKE log-log plot.
        tke_fft_interp = interp3((tke_fft.abs() ** 2).log(), p_query).exp().float()
        # Weights sum up to 1, so scale by the surface area of the radius-k sphere
        E_k = torch.matmul(tke_fft_interp, self.w) * (4 * torch.pi * k**2)

        # Sum the energies over all three dimensions to get the total energy
        return E_k
    
class LogTKESpectrumL2Distance(nn.Module):
    """
    Estimate the L2 distance between the log-TKE spectrum functions E(k) of two
    turbulent flows with Gauss-Legendre integration.
    """

    def __init__(self, tke_spectrum: nn.Module, n: int = 64):
        """
        Arguments
        ---------
        tke_spectrum: Module to compute the TKE spectrum of a velocity field
        n: Number of nodes for Gauss-Legendre integration
        """

        super().__init__()

        self.tke_spectrum = tke_spectrum
        self.n = n

        legendre_nodes, legendre_weights = roots_legendre(n)
        legendre_nodes = torch.tensor(legendre_nodes)
        legendre_weights = torch.tensor(legendre_weights)

        self.register_buffer("legendre_nodes", legendre_nodes.float())
        self.register_buffer("legendre_weights", legendre_weights.float())

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor, u_mean = None ):
        # assume u_a is prediction and u_b is target
        # assume u_a in shape b c t d h w. also assume batch size is 1
        u_a = u_a[0, :3].permute(1, 0, 2, 3, 4) # take first batch and velocity channels
        u_b = u_b[0, :3].permute(1, 0, 2, 3, 4)
        if u_mean == None:
            u_mean = torch.mean(u_b, dim=0)

        # Ensure that we don't also pass the pressure by accident
        # assuming u in shape t c d h w
        assert u_a.shape[-4] == 3
        assert u_b.shape[-4] == 3
        assert u_mean.shape[-4] == 3

        # Ensure that all velocity fields have the same spatial dimensions
        assert u_a.shape[-3:] == u_b.shape[-3:]
        assert u_a.shape[-3:] == u_mean.shape[-3:]

        # Linearly transform the Legendre nodes from [-1, 1] to the valid range of
        # frequencies k
        k_min = 1.0
        k_max = float((min(u_a.shape[-3:]) - 1) // 2)
        slope = (k_max - k_min) / 2
        k = slope * self.legendre_nodes + ((k_max - k_min) / 2 + k_min)

        # Compute the log-TKE-spectra
        log_tke_a = self.tke_spectrum(u_a - u_mean, k).log()
        log_tke_b = self.tke_spectrum(u_b - u_mean, k).log()
        
        # Compute the pairwise distances between any two log-TKE-spectra
        D = slope * torch.einsum(
            "ijk, k -> ij",
            (log_tke_a[:, None] - log_tke_b[None]) ** 2,
            self.legendre_weights,
        )
        D = torch.sqrt(D)
        
        # D is pairwise, so diagonals have distance between same timesteps. 
        D = torch.mean(torch.diagonal(D))
        
        return D, log_tke_a, log_tke_b, k
        