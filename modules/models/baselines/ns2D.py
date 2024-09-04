import torch
import torch.nn.functional as F

import lightning as L
from einops import rearrange
from modules.models.baselines.fno import FNO2d_cond
from modules.models.baselines.unet import Unet2D_cond
from modules.models.baselines.resnet import ResNet, DilatedBasicBlock, BasicBlock

# LightningModule for training Unet, FNO, ResNet, Dilated ResNet baselines

class NS2DModule(L.LightningModule):
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
            self.model = FNO2d_cond(**modelconfig[self.name])
        elif self.name == "unet":
            self.model = Unet2D_cond(**modelconfig[self.name])
        elif self.name == "resnet":
            self.model = ResNet(block=BasicBlock, **modelconfig[self.name])
        elif self.name == "dil_resnet":
            self.model = ResNet(block=DilatedBasicBlock, **modelconfig[self.name])
        else:
            raise ValueError("Model not recognized")
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.trainconfig = trainconfig

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

    def forward(self, x, cond=None):
        '''
        x : torch.Tensor
            input function a defined on the input domain 
            shape (batch, x, nx, ny) 
        cond: torch.Tensor
            conditional information
            shape (batch, cond_channels)
        '''
        if self.name == "dpot":
            x = x.unsqueeze(1) # b 1 x y c
            x = rearrange(x, 'b t x y c -> b x y t c')
            out = self.model(x)[0] 
            out = rearrange(out, 'b x y t c -> b t x y c') # b 1 x y c
            out = out.squeeze() # b x y c
            x = rearrange(x, 'b x y t c -> b t x y c')
            x = x.squeeze() # b x y c

        else:
            x = rearrange(x, 'b x y c -> b c x y')
            out = self.model(x, cond)
            out = rearrange(out, 'b c x y -> b x y c')

        return out
    
    def get_data_labels(self, x, batch_size=None):
        # x in shape b t x y c

        if batch_size is None:
            batch_size = self.batch_size

        t_data = torch.randint(0, x.shape[1]-1, (batch_size,), device=x.device)
        batch_range = torch.arange(batch_size, device=x.device)
        t_labels = t_data + 1 

        data = x[batch_range, t_data]
        labels = x[batch_range, t_labels]

        return data, labels

    def training_step(self, batch, batch_idx):
 
        inputs = batch["x"] # in shape b t x y c
        cond = batch.get("cond", None)
        inputs = self.normalizer.normalize(inputs, cond)  # normalize inputs to [-1, 1]

        if isinstance(inputs, tuple):
            inputs, cond = inputs

        data, labels= self.get_data_labels(inputs)

        pred = self(data, cond)
        loss = F.mse_loss(pred, labels)

        self.log("train/mse_loss", loss, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True,)
        
        labels_denorm = self.normalizer.denormalize(labels)
        pred_denorm = self.normalizer.denormalize(pred)

        train_loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        self.log("train/l1_loss", train_loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)

        return loss 

    def validation_step(self, batch, batch_idx, eval=False):
        inputs = batch["x"] # b t x y c
        t = inputs.shape[1]
        
        cond = batch.get("cond", None)

        inputs = self.normalizer.normalize(inputs, cond)  # normalize inputs to [-1, 1]

        if isinstance(inputs, tuple):
            inputs, cond = inputs

        batch_size = 1 if eval else self.batch_size
        data, labels = self.get_data_labels(inputs, batch_size=batch_size)

        pred = self(data, cond)
        loss = F.mse_loss(pred, labels)
        
        labels_denorm = self.normalizer.denormalize(labels)
        pred_denorm = self.normalizer.denormalize(pred)
        loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        
        x_start = inputs[:, 0] # b x y c
        rollout_error = 0 

        all_errors = []
        all_preds = torch.zeros_like(inputs)
        all_preds[:, 0] = x_start

        for i in range(t-1):
            pred_step = self(x_start, cond)
            target = inputs[:, i+1]
            single_step_error = F.l1_loss(self.normalizer.denormalize(pred_step), self.normalizer.denormalize(target))

            all_errors.append(single_step_error.item())
            all_preds[:, i+1] = pred_step

            x_start = pred_step

        if eval:
            return all_errors, self.normalizer.denormalize(all_preds)
        
        rollout_error = F.l1_loss(self.normalizer.denormalize(all_preds), self.normalizer.denormalize(inputs))

        self.log("val/mse_loss", loss, prog_bar=True,
                      logger=True, on_step=False, on_epoch=True,)
        self.log("val/l1_loss", loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,)
        self.log("val/rollout_error", rollout_error, prog_bar=False,
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