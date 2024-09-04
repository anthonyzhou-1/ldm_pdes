import torch
import os
import torch.nn.functional as F
import numpy as np
import lightning as L
from einops import rearrange, repeat, reduce
from functools import partial
from tqdm import tqdm

from modules.modules.distributions import DiagonalGaussianDistribution, normal_kl
from modules.models.unet import UNetModel
from modules.models.transformer import DiT
from modules.modules.diffusion import make_beta_schedule, extract_into_tensor, noise_like
from modules.utils import default, count_params

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


##################################################################
# DDPM
##################################################################

class DDPM(L.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 model_config,
                 timesteps=100,
                 beta_schedule="linear",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 parameterization="eps",  # all assuming fixed variance schedules
                 base_learning_rate=1e-4,
                 scheduler_config = None,
                 dist=False,
                 batch_size=1,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps, x0, and v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size  
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size, self.image_size)

        self.channels = channels
        self.model = UNetModel(**model_config)
        self.learning_rate = base_learning_rate
        self.scheduler_config = scheduler_config
        self.batch_size = batch_size
        count_params(self.model, verbose=True)

        self.v_posterior = v_posterior

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.dist = dist

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', to_torch(betas))
        register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        # loss weight

        snr = to_torch(alphas_cumprod / (1 - alphas_cumprod))

        maybe_clipped_snr = snr.clone()

        if self.parameterization == 'eps':
            loss_weight = maybe_clipped_snr / snr
        elif self.parameterization == 'x0':
            loss_weight = maybe_clipped_snr
        elif self.parameterization == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)
        else:
            raise ValueError(f'unknown objective {self.parameterization}')

        register_buffer('loss_weight', loss_weight)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def predict_start_from_noise(self, x_t, t, noise):
        # predict x_0 from x_t and noise
        # During inference, x_0 is not known, so must be predicted to calculate q(x_{t-1} | x_t, x_0)

        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_v(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_posterior(self, x_start, x_t, t):
        # Calculate the distribution q(x_{t-1} | x_t, x_0)
        # Since q is assumed to be Gaussian, this is tractable

        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        # Calculate the distribution p_theta(x_{t-1} | x_t) 

        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_v(x, t=t, v=model_out)
        if clip_denoised:
            x_recon.clamp_(-1., 1.) # clamp the latent space to -1, 1

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps, leave=False):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=1, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1], image_size[2]),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        # Sample from the forward diffusion process q(x_t | x_0)

        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.parameterization}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract_into_tensor(self.loss_weight, t, loss.shape)
        loss = loss.mean()

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        x = batch # assume x is (b, t, m, c), c has u v p channels
        x = rearrange(x, 'b t m c -> b c t m')
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True,
                      sync_dist=self.dist, batch_size=self.batch_size)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False,
                 sync_dist=self.dist, batch_size=self.batch_size)

        if self.scheduler_config is not None:
            sch = self.lr_schedulers()
            sch.step()

            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False,
                     sync_dist=self.dist, batch_size=self.batch_size)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=self.dist, batch_size=self.batch_size)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt



##################################################################
# LDM
##################################################################

class ACDM(DDPM):
    """main class"""
    def __init__(self,
                 normalizer = None,
                 *args, **kwargs):

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(*args, **kwargs)
        self.normalizer = normalizer

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def register_schedule(self,
                          beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)
    
    @torch.no_grad()
    def get_input(self, batch, data_inference=None):
        x = batch['x'] # b t nx ny c
        cond = batch.get('cond', None) # b 1
        batch_range = torch.arange(x.shape[0], device=self.device)

        if data_inference is None:
            t_max = x.shape[1]
            t_cond = torch.randint(0, t_max-1, (x.shape[0],), device=self.device).long()
            t_data = t_cond + 1

            d = x[batch_range, t_data]
            c = x[batch_range, t_cond]

        else:
            # assume data_inference is tuple with data and condition
            d = data_inference[0]
            c = data_inference[1]

        if self.normalizer is not None:
            d, cond = self.normalizer.normalize(d, cond)
            c = self.normalizer.normalize(c)

        d = rearrange(d, 'b nx ny c -> b c nx ny')
        c = rearrange(c, 'b nx ny c -> b c nx ny')
        data = torch.cat([c, d], dim=1)
        out = [data, cond]
        return out

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        # x has both conditioning (time t) and data (time t+1), shape [b, 2*c, nx, ny]
        # c is buoyancy factor, shape (b, 1)

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc, leave=False):
            rec = zd.to(self.device)
            denoise_row.append(rec)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W

        return denoise_row

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, y=cond) # use cond as global conditioning
        loss_dict = {}

        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.parameterization}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract_into_tensor(self.loss_weight, t, loss.shape)

        log_prefix = 'train' if self.training else 'val'

        loss_total = loss
        loss_dict.update({f'{log_prefix}/loss': loss_total.clone().detach().mean()})

        return loss_total.mean(), loss_dict


    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False):
        # x is assumed to have conditioning and data, shape [b, 2*c, nx, ny]
        # c is buoyancy factor, shape (b, 1)

        t_in = t
        model_out = self.model(x, t_in, y=c) # get prediction of noise from model

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out) # predict x_0 from x_t and noise
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_v(x, t=t, v=model_out)
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t) # calculate q(x_{t-1} | x_t, x_0)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, repeat_noise=False, return_x0=False, temperature=1., noise_dropout=0):
        # x is assumed to have conditioning and data, shape [b, 2*c, nx, ny]
        # c is buoyancy factor, shape (b, 1)

        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0,)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, c, cond, return_intermediates=False):
        # c is assumed to only have conditioning, shape [b, c, nx, ny]
        # cond is assumed to have global conditioning, shape [b, 1]

        log_every_t = self.log_every_t
        device = self.betas.device
        shape = c.shape

        # get the noise for the data
        dNoisy = torch.randn(shape, device=device)

        intermediates = [dNoisy]
        timesteps = self.num_timesteps

        # Make iterator from T to 0
        iterator = reversed(range(0, timesteps))

        for i in iterator:
            b = c.shape[0]
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            # get noisy conditioning and time t
            cNoisy = self.q_sample(x_start=c, t=ts)

            # get img to denoise
            img = torch.cat([cNoisy, dNoisy], dim=1)

            # get denoised image from x_T to x_T-1
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,)

            dNoisy = img[:, c.shape[1]:]  # get the denoised data
            # log intermediate images
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(dNoisy)

        if return_intermediates:
            return dNoisy, intermediates
        return dNoisy

    @torch.no_grad()
    def sample(self, c, cond, return_intermediates=False):

        return self.p_sample_loop(c,
                                  cond,
                                  return_intermediates=return_intermediates, 
                                  )

    @torch.no_grad()
    def sample_log(self, c, cond):

        samples, intermediates = self.sample(c=c, cond=cond, return_intermediates=True)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, 
                   batch, 
                   N=1, 
                   n_row=1, 
                   sample=True, 
                   plot_denoise_rows=True, 
                   plot_diffusion_rows=True, 
                   horizon=48,
                   verbose=False,
                   ):
        true_rollout = batch['x']
        cond = batch.get('cond', None)

        if self.normalizer is not None:
            true_rollout, cond = self.normalizer.normalize(true_rollout, cond)

        total_log = dict()
        pred_rollout = torch.zeros_like(true_rollout)
        samples = true_rollout[:, 0] # b nx ny c
        pred_rollout[:, 0] = samples

        for timestep in tqdm(range(horizon-1), desc='Timestep', leave=False):
            log = dict()

            log['samples'] = samples # conditioning, x_t
            log['inputs'] = true_rollout[:, timestep+1]  # true label, x_t+1

            if plot_diffusion_rows:
                # get diffusion row
                diffusion_row = list()
                z_start = samples[:n_row]
                for t in range(self.num_timesteps):
                    if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                        t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                        t = t.to(self.device).long()
                        noise = torch.randn_like(z_start) # Gets random noise
                        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise) # gets noised latent z at time t
                        diffusion_row.append(z_noisy)

                diffusion_row = torch.stack(diffusion_row)  # n_log_step, H W C
                log["diffusion_row"] = diffusion_row # logs forward process from x_0 (data sample) to x_T (fully noised)

            if sample:
                # get denoised row
                samples = rearrange(samples, 'b h w c -> b c h w')
                samples, z_denoise_row = self.sample_log(c=samples, cond=cond) # get denoised samples, x_t+1
                samples = rearrange(samples, 'b c h w -> b h w c')
                # plot intermediate denoised samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    log["denoise_row"] = denoise_grid # logs reverse process from x_0 (generated sample) to x_T (fully noised)

            pred_rollout[:, timestep+1] = samples
            if verbose:
                total_log[str(timestep)] = log

        total_log["samples"] = self.normalizer.denormalize(pred_rollout)
        total_log["inputs"] = self.normalizer.denormalize(true_rollout)
        return total_log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        if self.scheduler_config is not None:
            if self.scheduler_config["scheduler"] == "cosine":
                effective_batch_size = self.scheduler_config["batch_size"] * self.scheduler_config["accumulate_grad_batches"]
                max_epochs = self.scheduler_config["max_epochs"]
                dataset_size = self.scheduler_config["dataset_size"]
                T_max = max_epochs * (dataset_size // effective_batch_size  + 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,
                                                                      T_max=T_max,)
                print(f"Using CosineAnnealingLR with T_max={T_max}")
        
            return [opt], [scheduler]
        return opt