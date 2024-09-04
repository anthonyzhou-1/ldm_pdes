import torch
import torch.nn.functional as F
import lightning as L
from modules.models.ae.gino_ae import Encoder, Decoder
from modules.modules.distributions import DiagonalGaussianDistribution
from modules.losses.loss import LPIPSWithDiscriminator, KL_Loss
from einops import rearrange, repeat

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

        self.encoder = Encoder(**aeconfig["encoder"])
        self.decoder = Decoder(**aeconfig["decoder"])

        if lossconfig["loss"]["disc_weight"] > 0:
            self.discriminator = Encoder(**lossconfig["discriminator"])
        else:
            self.discriminator = None
            print("No discriminator used")

        if lossconfig["loss"]["perceptual_weight"] > 0:
            raise NotImplementedError("LPIPS not implemented for unstructured AE")
        else:
            self.lpips = None
        
        self.loss = LPIPSWithDiscriminator(discriminator=self.discriminator,
                                           lpips=self.lpips,
                                           **lossconfig["loss"])
        assert aeconfig["double_z"] # need to double the latent dimension to sample mean and std
        self.trainconfig = trainconfig
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

        latent_grid = self.get_latent_grid(aeconfig["latent_grid_size"])
        latent_grid_disc = self.get_latent_grid(aeconfig["disc_latent_grid_size"])

        self.register_buffer("latent_grid", latent_grid)
        self.register_buffer("latent_grid_disc", latent_grid_disc)

        self.automatic_optimization = False
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

    def get_latent_grid(self, N):
        xx = torch.linspace(0, 1, N)
        yy = torch.linspace(0, 1, N)
        tt = torch.linspace(0, 1, N)

        xx, yy, tt = torch.meshgrid(xx, yy, tt, indexing='ij')
        latent_queries = torch.stack([xx, yy, tt], dim=-1)
        
        return latent_queries.unsqueeze(0)

    def encode(self, x, pos, latent_queries, pad_mask=None):
        h = self.encoder(x, pos, latent_queries, pad_mask=pad_mask)
        #moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z, latent_queries, pos, pad_mask=None):
        #z = self.post_quant_conv(z)
        dec = self.decoder(z, latent_queries, pos, pad_mask=pad_mask)
        return dec

    def forward(self, x, pos, latent_queries, pad_mask = None, sample_posterior=True):
        posterior = self.encode(x, pos, latent_queries, pad_mask=pad_mask)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, latent_queries, pos, pad_mask=pad_mask)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        # inputs in shape b t m c, pos in shape b t m 3
        inputs = batch['x']
        pos = batch['pos']
        pad_mask = batch.get('pad_mask', None)

        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b n -> b 1 n 1')
            inputs = inputs * pad_mask  # make sure padding values are still zero after normalization

        latent_queries = self.latent_grid

        if self.discriminator is not None:
            opt_0, opt_1 = self.optimizers()
            sch0, sch1 = self.lr_schedulers()
        
        else:
            opt_0 = self.optimizers()
            sch0 = self.lr_schedulers()

        reconstructions, posterior = self(inputs, pos, latent_queries, pad_mask=pad_mask) # reconstructions are in normalized space [-1, 1]

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train", normalizer=self.normalizer,
                                        disc_pos=pos, disc_latent=self.latent_grid_disc, pad_mask=pad_mask)
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_epoch=True, sync_dist=self.dist)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_epoch=False, sync_dist=self.dist)
        self.manual_backward(aeloss/self.accumulation_steps)

        # train the discriminator
        if self.discriminator is not None:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", normalizer=self.normalizer,
                                                disc_pos=pos, disc_latent=self.latent_grid_disc, pad_mask=pad_mask)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.dist)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.dist)
            self.manual_backward(discloss/self.accumulation_steps)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt_0.step()
            sch0.step()
            opt_0.zero_grad()

            if self.discriminator is not None:
                opt_1.step()
                sch1.step()
                opt_1.zero_grad()

            self.log("lr", sch0.get_last_lr()[0], sync_dist=self.dist)

    def validation_step(self, batch, batch_idx, eval=False):
        # inputs in shape b t m c, pos in shape b t m 3
        inputs = batch['x']
        pos = batch['pos']
        pad_mask = batch.get('pad_mask', None)

        inputs = self.normalizer.normalize(inputs)  # normalize inputs

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b n -> b 1 n 1')
            inputs = inputs * pad_mask  # make sure padding values are zero after normalization

        latent_queries = self.latent_grid

        reconstructions, posterior = self(inputs, pos, latent_queries, pad_mask=pad_mask)

        if eval:
            return reconstructions

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", normalizer=self.normalizer,
                                        disc_pos=pos, disc_latent=self.latent_grid_disc, pad_mask=pad_mask)

        # train the discriminator
        if self.discriminator is not None:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", normalizer=self.normalizer,
                                            disc_pos=pos, disc_latent=self.latent_grid_disc, pad_mask=pad_mask)
        
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=self.dist, on_epoch=True)
        self.log_dict(log_dict_ae, on_epoch=True, sync_dist=self.dist)
        if self.discriminator is not None:
            self.log_dict(log_dict_disc, on_epoch=True, sync_dist=self.dist)

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        
        if self.discriminator is not None:
            opt_disc = torch.optim.Adam(list(self.discriminator.parameters()),
                                        lr=lr, betas=(0.5, 0.9))
            
        effective_batch_size = self.batch_size * self.accumulation_steps
        if self.trainconfig["scheduler"] == "OneCycle":
            scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_ae,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)
            if self.discriminator is not None:
                scheduler_disc = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_disc,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)

        elif self.trainconfig["scheduler"] == "Cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_ae,
                                                                      T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
            if self.discriminator is not None:
                scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_disc,
                                                                        T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
        else:
            scheduler_ae = None
            scheduler_disc = None

        if self.discriminator is not None:
            return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
        else:
            return [opt_ae], [scheduler_ae]
    
    def get_last_layer(self):
        return self.decoder.gino_decoder.projection.fcs[-1].weight

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

        self.encoder = Encoder(**aeconfig["encoder"])
        self.decoder = Decoder(**aeconfig["decoder"])
        
        assert aeconfig["double_z"] # need to double the latent dimension to sample mean and std
        self.trainconfig = trainconfig
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

        self.loss = KL_Loss(**lossconfig)

        latent_grid = self.get_latent_grid(aeconfig["latent_grid_size"])
        self.register_buffer("latent_grid", latent_grid)

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

    def get_latent_grid(self, N):
        xx = torch.linspace(0, 1, N)
        yy = torch.linspace(0, 1, N)
        tt = torch.linspace(0, 1, N)

        xx, yy, tt = torch.meshgrid(xx, yy, tt, indexing='ij')
        latent_queries = torch.stack([xx, yy, tt], dim=-1)
        
        return latent_queries.unsqueeze(0)

    def encode(self, x, pos, latent_queries, pad_mask=None):
        h = self.encoder(x, pos, latent_queries, pad_mask=pad_mask)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z, latent_queries, pos, pad_mask=None):
        #z = self.post_quant_conv(z)
        dec = self.decoder(z, latent_queries, pos, pad_mask=pad_mask)
        return dec

    def forward(self, x, pos, latent_queries, pad_mask = None, sample_posterior=True):
        posterior = self.encode(x, pos, latent_queries, pad_mask=pad_mask)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, latent_queries, pos, pad_mask=pad_mask)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        # inputs in shape b t m c, pos in shape b t m 3
        inputs = batch['x']
        pos = batch['pos']
        pad_mask = batch.get('pad_mask', None)

        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b n -> b 1 n 1')
            inputs = inputs * pad_mask  # make sure padding values are still zero after normalization

        latent_queries = self.latent_grid

        reconstructions, posterior = self(inputs, pos, latent_queries, pad_mask=pad_mask) # reconstructions are in normalized space 

        loss, log_dict = self.loss(inputs, reconstructions, posterior,
                                   split="train", normalizer=self.normalizer, pad_mask=pad_mask)

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=self.dist)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=self.dist)

        sch = self.lr_schedulers()
        sch.step()

        return loss 

    def validation_step(self, batch, batch_idx, eval=False):
        # inputs in shape b t m c, pos in shape b t m 3
        inputs = batch['x']
        pos = batch['pos']
        pad_mask = batch.get('pad_mask', None)

        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b n -> b 1 n 1')
            inputs = inputs * pad_mask  # make sure padding values are still zero after normalization

        latent_queries = self.latent_grid

        reconstructions, posterior = self(inputs, pos, latent_queries, pad_mask=pad_mask) # reconstructions are in normalized space 

        if eval:
            return reconstructions

        loss, log_dict = self.loss(inputs, reconstructions, posterior,
                                   split="val", normalizer=self.normalizer, pad_mask=pad_mask)

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=self.dist)

        return loss 

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
