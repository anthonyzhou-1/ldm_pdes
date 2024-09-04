import torch
import lightning as L
from modules.models.ae.cnn_ae import CNN_Decoder, CNN_Encoder
from modules.modules.distributions import DiagonalGaussianDistribution
from modules.models.discriminator import NLayerDiscriminator3D
from modules.losses.loss import LPIPSWithDiscriminator, KL_Loss
from modules.losses.lpips import LPIPS_DPOT
from einops.layers.torch import Rearrange

class Autoencoder(L.LightningModule):
    def __init__(self,
                 aeconfig,
                 lossconfig,
                 trainconfig,
                 normalizer=None,
                 ckpt_path=None,
                 batch_size = 1,
                 accumulation_steps = 1,
                 ):
        super().__init__()

        self.encoder = CNN_Encoder(**aeconfig["encoder"])
        self.decoder = CNN_Decoder(**aeconfig["decoder"])
        if "attn_resolutions" in lossconfig["discriminator"]:
            self.discriminator = CNN_Encoder(**lossconfig["discriminator"])
        else:
            self.discriminator = NLayerDiscriminator3D(**lossconfig["discriminator"])

        self.to_conv_shape = Rearrange('b t h w c -> b c t h w')
        self.to_input_shape = Rearrange('b c t h w -> b t h w c')

        if lossconfig["loss"]["perceptual_weight"] > 0:
            self.lpips = LPIPS_DPOT(**lossconfig["lpips"]).eval()
        else:
            self.lpips = None

        self.loss = LPIPSWithDiscriminator(discriminator=self.discriminator,
                                           lpips=self.lpips,
                                           **lossconfig["loss"])
        assert aeconfig["encoder"]["double_z"] 
        self.trainconfig = trainconfig
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.dist = trainconfig.get("dist", False)

        self.automatic_optimization = False
        self.save_hyperparameters()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, cond=None):
        h = self.encoder(x=x, cond=cond)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z, cond=None):
        dec = self.decoder(z=z, cond=cond)
        return dec

    def forward(self, x, sample_posterior=True, cond=None):
        # expects x in shape [batch, c, t, h, w]    
        posterior = self.encode(x=x, cond=cond)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z=z, cond=cond)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = batch["x"]
        cond = batch.get("cond", None)

        opt_0, opt_1 = self.optimizers()
        sch0, sch1 = self.lr_schedulers()

        inputs = self.normalizer.normalize(inputs, cond)
        if isinstance(inputs, tuple):
            inputs, cond = inputs

        inputs = self.to_conv_shape(inputs)
        reconstructions, posterior = self(inputs, cond=cond)

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, cond=cond,
                                        last_layer=self.get_last_layer(), split="train", normalizer=self.normalizer)
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_epoch=True, sync_dist=self.dist)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_epoch=False, sync_dist=self.dist)

        self.manual_backward(aeloss/self.accumulation_steps)

        # train the discriminator
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, cond=cond,
                                            last_layer=self.get_last_layer(), split="train", normalizer=self.normalizer)

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.dist)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.dist)

        self.manual_backward(discloss/self.accumulation_steps)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt_0.step()
            sch0.step()
            opt_0.zero_grad()

            opt_1.step()
            sch1.step()
            opt_1.zero_grad()

            self.log("lr", sch0.get_last_lr()[0], sync_dist=self.dist)

    def validation_step(self, batch, batch_idx, eval=False):
        inputs = batch["x"]
        cond = batch.get("cond", None)

        inputs = self.normalizer.normalize(inputs, cond)

        if isinstance(inputs, tuple):
            inputs, cond = inputs

        inputs = self.to_conv_shape(inputs)
        reconstructions, posterior = self(inputs, cond=cond)

        if eval:
            reconstructions = self.to_input_shape(reconstructions)
            reconstructions = self.normalizer.denormalize(reconstructions)
            return reconstructions

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, cond=cond,
                                        last_layer=self.get_last_layer(), split="val", normalizer=self.normalizer)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, cond=cond,
                                            last_layer=self.get_last_layer(), split="val", normalizer=self.normalizer)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=self.dist, on_epoch=True)
        self.log_dict(log_dict_ae, sync_dist=self.dist, on_epoch=True)
        self.log_dict(log_dict_disc, sync_dist=self.dist, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(list(self.discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
            
        effective_batch_size = self.batch_size * self.accumulation_steps
        if self.trainconfig["scheduler"] == "OneCycle":
            scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_ae,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)

            scheduler_disc = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_disc,
                                                        max_lr=lr,
                                                        total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                        pct_start=self.trainconfig["pct_start"],)

        elif self.trainconfig["scheduler"] == "Cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_ae,
                                                                      T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
            scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_disc,
                                                                        T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
        else:
            scheduler_ae = None
            scheduler_disc = None

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
    

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
class AutoencoderKL(L.LightningModule):
    def __init__(self,
                 aeconfig,
                 lossconfig,
                 trainconfig,
                 normalizer=None,
                 ckpt_path=None,
                 batch_size = 1,
                 accumulation_steps = 1,
                 ):
        super().__init__()

        self.encoder = CNN_Encoder(**aeconfig["encoder"])
        self.decoder = CNN_Decoder(**aeconfig["decoder"])

        self.to_conv_shape = Rearrange('b t h w c -> b c t h w')
        self.to_input_shape = Rearrange('b c t h w -> b t h w c')

        self.lpips = None
        self.discriminator = None

        self.loss = KL_Loss(**lossconfig)
        self.trainconfig = trainconfig
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.dist = trainconfig.get("dist", False)

        self.save_hyperparameters()
        
        #print("Training with batch size", self.batch_size)
        #print("Training with accumulation steps", self.accumulation_steps)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, cond=None):
        h = self.encoder(x=x, cond=cond)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z, cond=None):
        dec = self.decoder(z=z, cond=cond)
        return dec

    def forward(self, x, sample_posterior=True, cond=None):
        # expects x in shape [batch, c, t, h, w]    
        posterior = self.encode(x=x, cond=cond)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z=z, cond=cond)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = batch["x"]
        cond = batch.get("cond", None)

        inputs = self.normalizer.normalize(inputs, cond)

        if isinstance(inputs, tuple):
            inputs, cond = inputs

        inputs = self.to_conv_shape(inputs)
        reconstructions, posterior = self(inputs, cond=cond)

        loss, log_dict = self.loss(inputs, reconstructions, posterior,
                                   split="train", normalizer=self.normalizer)

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=self.dist)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=self.dist)

        sch = self.lr_schedulers()
        sch.step()

        return loss 

    def validation_step(self, batch, batch_idx, eval=False):
        inputs = batch["x"]
        cond = batch.get("cond", None)

        inputs = self.normalizer.normalize(inputs, cond)

        if isinstance(inputs, tuple):
            inputs, cond = inputs

        inputs = self.to_conv_shape(inputs)
        reconstructions, posterior = self(inputs, cond=cond)

        if eval:
            reconstructions = self.to_input_shape(reconstructions)
            reconstructions = self.normalizer.denormalize(reconstructions)
            return reconstructions

        loss, log_dict = self.loss(inputs, reconstructions, posterior,
                                   split="val", normalizer=self.normalizer)
        
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=self.dist)

        return log_dict

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
            
        effective_batch_size = self.batch_size * self.accumulation_steps
        if self.trainconfig["scheduler"] == "OneCycle":
            scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_ae,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)

        elif self.trainconfig["scheduler"] == "Cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_ae,
                                                                      T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
        else:
            scheduler_ae = None

        return [opt_ae], [scheduler_ae]
