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


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

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
                 timesteps=1000,
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
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 base_learning_rate=1e-4,
                 scheduler_config = None,
                 cond_scale = None, 
                 rescaled_phi = None,
                 dist=False,
                 batch_size=1,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps, x0, and v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size  
        self.learn_sigma = model_config.get("learn_sigma", False)

        if self.learn_sigma:
            print("Learning sigma")

        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size, self.image_size)

        self.channels = channels
        self.model = DiffusionWrapper(model_config, conditioning_key)
        self.learning_rate = base_learning_rate
        self.scheduler_config = scheduler_config
        self.cond_scale = cond_scale
        self.rescaled_phi = rescaled_phi
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

class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 scale_factor=1.0,
                 normalizer = None,
                 use_embed = False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= kwargs['timesteps']

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        conditioning_key = "crossattn" if cond_stage_config["conditional"] else None
        print(conditioning_key)
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.normalizer = normalizer
        self.use_embed = use_embed
        self.ablate = False 

        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def register_schedule(self,
                          beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        aeconfig = config['aeconfig']
        lossconfig = config['lossconfig']
        trainconfig = config['training']
        pretrained_path = config['pretrained_path']

        from modules.models.ae.ae_grid import AutoencoderKL # only supports simple grid AE

        model = AutoencoderKL(aeconfig,
                            lossconfig,
                            trainconfig,
                            normalizer=self.normalizer)
            
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, map_location=self.device)["state_dict"])
            print(f"Autoencoder model loaded from path: {pretrained_path} onto device: {self.device}")
            model.discriminator = None # remove discriminator from model
            model.lpips = None # remove lpips from model
            model.loss = None 
        
        # Freeze autoencoder
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if config["conditional"]: 
            if self.use_embed:
                from transformers import AutoTokenizer, RobertaModel
                cache_path = "/pscratch/sd/a/ayz2/cache/roberta-base/"
                if os.path.isdir(cache_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(cache_path, local_files_only=True)
                    self.cond_stage_model = RobertaModel.from_pretrained(cache_path, local_files_only=True)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base") # On the fly. Might cause the script to hang.
                    self.cond_stage_model  = RobertaModel.from_pretrained("FacebookAI/roberta-base")
                print("Text embeddings model loaded")
                self.cond_stage_model.pooler = None # remove pooler from model
                    
            else:
                from modules.models.ae.cnn_ae import ConditionalEncoder
                model = ConditionalEncoder(config)
                self.cond_stage_model = model
                print(f"Conditional Encoder instantiated")
                
        else:
            print("No conditioning model instantiated")
            self.cond_stage_model = None

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.use_embed:
            # assume is roberta
            tokens = self.tokenizer(c, return_tensors='pt', padding=False) # assume training with batch size 1
            tokens = {k: v.to(self.device) for k, v in tokens.items()} # manually move to device
            embedding_mask = tokens['attention_mask'] # 1 where token is not padding, 0 where token is padding
            embedding = self.cond_stage_model(**tokens)["last_hidden_state"] # b, n, d_embed
            c = (embedding, embedding_mask)
        else:
            x, cond = c 
            c = self.cond_stage_model(x, cond)

        return c
    
    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        
        x = batch['x'] # b t d h w c
        pos = batch.get('pos', None) 
        pad_mask = batch.get('pad_mask', None) # b, n
        cond = batch.get('cond', None)
        mask = batch.get('mask', None) # b d h w

        if self.normalizer is not None:
            if cond is not None:
                x, cond = self.normalizer.normalize(x, cond)
            else:
                x = self.normalizer.normalize(x) # normalize input
            
        if mask is not None:
            mask = repeat(mask, 'b d h w -> b 1 d h w 1')
            x = x * mask

        encoder_posterior = self.encode_first_stage(x, pos, pad_mask=pad_mask, cond=cond) # encode x to posterior
        z = self.get_first_stage_encoding(encoder_posterior).detach() # sample from posterior

        if self.use_embed: # use text embeddings
            xc = batch.get('prompt', None) # list of text labels

        else:
            x0 = x[:, 0] # get first frame of x as conditioning info (b, nz, nx, ny, 4)
            x0 = rearrange(x0, 'b d h w c -> b c d h w') # b, c, d, h, w
            xc = (x0, cond)

        if force_c_encode:
            c = self.get_learned_conditioning(xc)
        else:
            c = xc

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z, pos, pad_mask=pad_mask, cond=cond)
            xrec = xrec * mask if mask is not None else xrec
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, pos, pad_mask=None, cond = None):
        # z is expected in b c n1 n2 n3
        z = 1. / self.scale_factor * z

        reconstructions = self.first_stage_model.decode(z, cond)
        reconstructions = rearrange(reconstructions, 'b c t d h w -> b t d h w c')

        if self.normalizer is not None:
            reconstructions = self.normalizer.denormalize(reconstructions)

        return reconstructions

    @torch.no_grad()
    def encode_first_stage(self, x, pos=None, pad_mask=None, cond=None):
        x = rearrange(x, 'b t d h w c -> b c t d h w') # 3d rearrangement
        return self.first_stage_model.encode(x, cond)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.cond_stage_trainable:
            c = self.get_learned_conditioning(c) # if trainable, do a forward pass to get conditioning
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False, cond_scale=None, rescaled_phi=None):
        if cond_scale is not None:
            x_recon = self.model(x_noisy, t, cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
        else:
            x_recon = self.model(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _get_denoise_row_from_list(self, samples, pos, pad_mask=None, cond=None, mask=None, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc, leave=False):
            rec = self.decode_first_stage(zd.to(self.device), pos, pad_mask=pad_mask, cond=cond)
            rec = rec * mask if mask is not None else rec
            denoise_row.append(rec)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W

        return denoise_row
    
    def _vb_terms_bpd(
            self, out, x_start, x_t, t, clip_denoised,
        ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        model_mean, _ , posterior_log_variance = self.p_mean_variance(
            x=x_t, c=None, t=t, clip_denoised=clip_denoised, out=out,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, posterior_log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)
        return kl

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, cond)

        if self.learn_sigma:
            B, C = x_noisy.shape[:2]
            assert model_out.shape == (B, C * 2, *x_noisy.shape[2:]), f"model out shape: {model_out.shape}, x_noisy shape: {x_noisy.shape}"
            model_out, model_var_values = torch.split(model_out, C, dim=1)

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

        if self.learn_sigma:
            # Learn the variance using the variational bound, but don't let it affect our mean prediction.
            frozen_out = torch.cat([model_out.detach(), model_var_values], dim=1)
            loss_vb = self._vb_terms_bpd(
                out=frozen_out,
                x_start=x_start,
                x_t=x_noisy,
                t=t,
                clip_denoised=self.clip_denoised,
            )
            loss_vb = self.num_timesteps / 1000.0 * loss_vb 

            loss_dict.update({f'{log_prefix}/loss_vb': loss_vb.clone().detach().mean()})
            loss_dict.update({f'{log_prefix}/loss_target': loss.clone().detach().mean()})

            loss_total = loss + loss_vb

            loss_dict.update({f'{log_prefix}/loss': loss_total.clone().detach().mean()})

        else:
            loss_total = loss
            loss_dict.update({f'{log_prefix}/loss': loss_total.clone().detach().mean()})

        return loss_total.mean(), loss_dict


    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, cond_scale=None, rescaled_phi=None, out=None):
        t_in = t
        if out is None:
            model_out = self.apply_model(x, t_in, c, cond_scale=cond_scale, rescaled_phi=rescaled_phi) # get prediction of noise from model
        else:
            model_out = out

        if self.learn_sigma:
            B, C = x.shape[:2]
            assert model_out.shape == (B, C * 2, *x.shape[2:]), f"model out shape: {model_out.shape}, x shape: {x.shape}"
            model_out, model_var_values = torch.split(model_out, C, dim=1)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            posterior_log_variance= frac * max_log + (1 - frac) * min_log
            posterior_variance = torch.exp(posterior_log_variance)

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

        if not self.learn_sigma:
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t) # calculate q(x_{t-1} | x_t, x_0)
        elif self.learn_sigma:
            model_mean, _, _ =  self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, repeat_noise=False, return_x0=False, temperature=1., noise_dropout=0.,
                 cond_scale=None, rescaled_phi=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
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
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, 
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              batch_size=None, x_T=None, start_T=None,
                              log_every_t=None, cond_scale=None, rescaled_phi=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps

        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        
        # create x_T (fully noised) if not given
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T

        intermediates = []

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            # get denoised img from x_T to x_T-1 as well as prediction of x_0
            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            cond_scale=cond_scale, rescaled_phi=rescaled_phi)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, timesteps=None, 
                        mask=None, x0=None,  start_T=None, log_every_t=None, cond_scale=None, rescaled_phi=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]

        # create x_T (fully noised) if not given
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]

        # get timesteps if not given
        if timesteps is None:
            timesteps = self.num_timesteps

        # get start time if not given
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        # Make iterator from T to 0
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps, leave=False)

        # Use the values of x0 at masked points
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:] == mask.shape[2:]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            # get denoised image from x_T to x_T-1
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                cond_scale=cond_scale, rescaled_phi=rescaled_phi)
            
            # add original image at masked points if mask is given
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            # log intermediate images
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=1, return_intermediates=False, x_T=None,
               timesteps=None, mask=None, x0=None, shape=None, cond_scale=None, rescaled_phi=None):
        
        if shape is None:
            shape = (batch_size, self.channels, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3])
        
        cond_scale = default(cond_scale, self.cond_scale)
        rescaled_phi = default(rescaled_phi, self.rescaled_phi)

        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, 
                                  x_T=x_T,
                                  timesteps=timesteps, 
                                  mask=mask, x0=x0,
                                  cond_scale=cond_scale, rescaled_phi=rescaled_phi)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, cond_scale=None, rescaled_phi=None):

        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,
                                                 cond_scale=cond_scale, rescaled_phi=rescaled_phi)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, 
                   batch, 
                   N=1, 
                   n_row=1, 
                   sample=True, 
                   plot_denoise_rows=True, 
                   plot_progressive_rows=False,
                   plot_diffusion_rows=True, 
                   cond_scale=None,
                   rescaled_phi=None,
                   ):
        
        pos = batch.get('pos', None)
        pad_mask = batch.get('pad_mask', None)
        cond = batch.get('cond', None)
        mask = batch.get('mask', None)

        if mask is not None:
            mask = repeat(mask, 'b d h w -> b 1 d h w 1')
        
        if cond is not None:
            cond = self.normalizer.normalize_cond(cond)

        cond_scale = default(cond_scale, self.cond_scale)
        rescaled_phi = default(rescaled_phi, self.rescaled_phi)

        log = dict()

        # get latent vector z, latent conditioning c, sample x, reconstruction xrec, and original conditioning xc
        z, c, x, xrec, xc = self.get_input(batch, 
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        
        N = min(x.shape[0], N) # num samples
        n_row = min(x.shape[0], n_row) # num rows when plotting
        log["inputs"] = self.normalizer.denormalize(x) * mask if mask is not None else self.normalizer.denormalize(x)
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            log["conditioning"] = xc

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start) # Gets random noise
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise) # gets noised latent z at time t
                    decoded = self.decode_first_stage(z_noisy, pos, pad_mask=pad_mask, cond=cond)
                    decoded = decoded * mask if mask is not None else decoded
                    diffusion_row.append(decoded) 

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, T D H W C
            log["diffusion_row"] = diffusion_row # logs forward process from x_0 (data sample) to x_T (fully noised)

        if sample:
            # get denoised row
            samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, cond_scale=cond_scale, rescaled_phi=rescaled_phi) # get denoised latent
            x_samples = self.decode_first_stage(samples, pos, pad_mask=pad_mask, cond=cond) # decode denoised latent into x_sample
            x_samples = x_samples * mask if mask is not None else x_samples
            log["samples"] = x_samples

            # plot intermediate denoised samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row, pos, pad_mask, cond, mask)
                log["denoise_row"] = denoise_grid # logs reverse process from x_0 (generated sample) to x_T (fully noised)

        if plot_progressive_rows:
            img, progressives = self.progressive_denoising(c,
                                                            shape=(self.channels, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                            batch_size=N,
                                                            cond_scale=cond_scale,
                                                            rescaled_phi=rescaled_phi)
            prog_row = self._get_denoise_row_from_list(progressives, pos, pad_mask, cond, mask)
            log["progressive_row"] = prog_row # logs reverse process from x_T (fully noised) to x_0 (generated sample)

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=200, gamma=0.85)
        print(f"Using StepLR")
        
        return [opt], [scheduler]


##################################################################
# Wrapper
##################################################################


class DiffusionWrapper(L.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        if "attention_resolutions" in diff_model_config.keys():
            self.diffusion_model = UNetModel(**diff_model_config)
        else:
            self.diffusion_model = DiT(**diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, context=None, cond_scale=None, rescaled_phi=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'crossattn':
            out = self.diffusion_model(x, t, context=context)
            if cond_scale is not None:
                out = self.diffusion_model.forward_with_cond_scale(x, t, cond_scale=cond_scale, rescaled_phi=rescaled_phi, context=context)
        else:
            raise NotImplementedError()

        return out