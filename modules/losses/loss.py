import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from modules.models.ddpm import mean_flat

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
            rec_loss = torch.abs(normalizer.denormalize(inputs) - normalizer.denormalize(reconstructions)) 
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # shape [b, c, nt, nx, ny]

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
                mask = torch.ones(b, 1, nt, nx, ny, device=inputs.device) # no mask for lpips

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
            mask = repeat(mask, 'b nx ny -> b 1 1 nx ny')
            rec_loss = rec_loss * (1-mask) # propagate only the error where mask is zero

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
        