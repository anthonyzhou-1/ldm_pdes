import torch
import torch.nn.functional as F

import lightning as L
from einops import rearrange, repeat
from modules.modules.fno_module import FNO
from modules.models.baselines.unet3D import Unet3D
from modules.models.baselines.resnet3D import ResNet, DilatedBasicBlock
from modules.models.baselines.factformer.factformer import FactorizedTransformer
from modules.losses.loss import ScaledLpLoss
from einops.layers.torch import Rearrange
from modules.losses.loss import LogTKESpectrumL2Distance, TurbulentKineticEnergySpectrum

# LightningModule for training FNO3D, Unet3D, FactFormer3D baselines

class Turb3DModule(L.LightningModule):
    def __init__(self,
                 modelconfig,
                 trainconfig,
                 normalizer=None,
                 batch_size=1,
                 accumulation_steps=1,
                 ckpt_path=None
                 ):
        super().__init__()
        self.name = modelconfig['name']
        if self.name == "fno":
            self.model = FNO(**modelconfig[self.name])
        elif self.name == "unet":
            self.model = Unet3D(**modelconfig[self.name])
        elif self.name == "dil_resnet":
            self.model = ResNet(block=DilatedBasicBlock, **modelconfig[self.name])
        elif self.name == "factformer":
            self.model = FactorizedTransformer(**modelconfig[self.name])
            self.init_pos_lst(resolution=modelconfig["resolution"])
        else:
            raise ValueError("Model not recognized")
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.trainconfig = trainconfig
        self.criterion = ScaledLpLoss()
        self.to_conv = Rearrange('b d h w c -> b c d h w')
        self.to_input = Rearrange('b c d h w -> b d h w c')

        self.tke_loss = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())

        self.save_hyperparameters()

        print("Training with batch size", self.batch_size)
        print("Training with accumulation steps", self.accumulation_steps)

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
    
    def init_pos_lst(self, resolution):
        d = torch.linspace(0, 1, resolution[0]).unsqueeze(-1)   
        h = torch.linspace(0, 1, resolution[1]).unsqueeze(-1)   
        w = torch.linspace(0, 1, resolution[2]).unsqueeze(-1)   
        self.register_buffer("d", d)
        self.register_buffer("h", h)
        self.register_buffer("w", w)

    def forward(self, x):
        '''
        x : torch.Tensor
            input function a defined on the input domain 
            shape (batch, x, nx, ny) 
        '''
        if self.name == "factformer":
            pos_lst = [self.d, self.h, self.w]
            out = self.model(x, pos_lst)
        else:
            x = self.to_conv(x)
            out = self.model(x)
            out = self.to_input(out)

        return out
    
    def get_data_labels(self, x, batch_size=None):
        # x in shape b t d h w c

        if batch_size is None:
            batch_size = self.batch_size

        t_data = torch.randint(0, x.shape[1]-1, (batch_size,), device=x.device)
        batch_range = torch.arange(batch_size, device=x.device)
        t_labels = t_data + 1 

        data = x[batch_range, t_data]
        labels = x[batch_range, t_labels]

        return data, labels

    def training_step(self, batch, batch_idx):
        inputs = batch["x"] # in shape b t d h w c
        mask = batch["mask"]  # in shape b d h w
        inputs = self.normalizer.normalize(inputs) 
        data, labels= self.get_data_labels(inputs)

        if mask is not None:
            mask = repeat(mask, 'b d h w -> b d h w c', c=inputs.shape[-1])
            data = data * mask # zero out data after normalization
            labels = labels * mask # zero out labels after normalization
        
        pred = self(data)
        pred = pred * mask # propagate loss only where mask is one
        loss = self.criterion(pred, labels) 

        self.log("train/rel_l2_loss", loss, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True,)
        
        labels_denorm = self.normalizer.denormalize(labels)
        pred_denorm = self.normalizer.denormalize(pred)

        train_loss_denorm = self.criterion(labels_denorm*mask, pred_denorm*mask) # single step error 
        self.log("train/rel_l2_loss_denorm", train_loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)

        return loss 

    def validation_step(self, batch, batch_idx, eval=False):
        inputs = batch["x"] # in shape b t d h w c
        t = inputs.shape[1]
        mask = batch["mask"]  # in shape b d h w
        inputs = self.normalizer.normalize(inputs) 
        data, labels= self.get_data_labels(inputs)

        if mask is not None:
            mask = repeat(mask, 'b d h w -> b d h w c', c=inputs.shape[-1])
            data = data * mask # zero out data after normalization
            labels = labels * mask # zero out labels after normalization

        pred = self(data)
        pred = pred * mask # propagate loss only where mask is one
        loss = self.criterion(pred, labels) 
        
        labels_denorm = self.normalizer.denormalize(labels)
        pred_denorm = self.normalizer.denormalize(pred)
        loss_denorm = self.criterion(pred_denorm*mask, labels_denorm*mask) # single step error 
        
        x_start = inputs[:, 0] # b d h w c

        all_errors = []
        all_preds = torch.zeros_like(inputs)
        all_preds[:, 0] = x_start*mask

        for i in range(t-1):
            pred_step = self(x_start*mask)
            target = inputs[:, i+1]

            pred_step_denorm = self.normalizer.denormalize(pred_step)*mask
            target_step_denorm = self.normalizer.denormalize(target)*mask

            single_step_error = self.criterion(pred_step_denorm, target_step_denorm)
            all_errors.append(single_step_error)
            all_preds[:, i+1] = pred_step_denorm

            x_start = pred_step

        all_errors = torch.stack(all_errors)

        if eval:
            return all_errors, all_preds # b t d h w c
        
        rollout_error = all_errors.mean()

        reconstructions = rearrange(all_preds, 'b t d h w c -> b c t d h w')
        inputs = rearrange(inputs * mask.unsqueeze(1), 'b t d h w c -> b c t d h w')
        tke_losses = []
        for i in range(4):
            idx_start = i*24
            tke_loss, _, _, _ = self.tke_loss(reconstructions[:, :, :, idx_start:idx_start+24], inputs[:, :, :, idx_start:idx_start+24])
            tke_losses.append(tke_loss)
        mean_tke_loss = torch.mean(torch.stack(tke_losses))

        self.log("val/rel_l2_loss", loss, prog_bar=True,
                      logger=True, on_step=False, on_epoch=True,)
        self.log("val/rel_l2_loss_denorm", loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)
        self.log("val/rollout_error", rollout_error, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)
        self.log("val/tke_loss", mean_tke_loss, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)
        
        return loss

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(self.model.parameters(),
                                    lr=lr,)
            
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