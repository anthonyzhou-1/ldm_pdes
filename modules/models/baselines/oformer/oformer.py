import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import lightning as L
from modules.models.baselines.oformer.decoder_module import IrregSTDecoder2D
from modules.models.baselines.oformer.encoder_module import IrregSTEncoder2D

######################
# Module
######################

class OFormerModule(L.LightningModule):
    def __init__(self,
                 modelconfig,
                 trainconfig,
                 normalizer=None,
                 batch_size=1,
                 accumulation_steps=1,
                 ckpt_path=None
                 ):
        super().__init__()

        self.encoder = IrregSTEncoder2D(**modelconfig["encoder"])
        self.decoder = IrregSTDecoder2D(**modelconfig["decoder"])
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.trainconfig = trainconfig

        self.dist = trainconfig['dist']

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

    def forward(self, x, pos):
        z = self.encoder(x, pos)
        pred = self.decoder(z, pos, pos)
        return pred  
    
    def get_graph_data_labels(self, x, y, pos, edge_idx) -> Data:
        # assume x, y in shape m c
        # pos in shape m 2
        # edge_idx in shape 2 e

        input_distances = torch.norm(x[edge_idx[0]] - x[edge_idx[1]], dim=1, p=1).unsqueeze(1)
        edge_distances = torch.norm(pos[edge_idx[0]] - pos[edge_idx[1]], dim=1, p=1).unsqueeze(1)

        edge_attr = torch.cat((input_distances, edge_distances), dim=1)

        graph = Data(x=x, edge_index=edge_idx, pos=pos, y=y, edge_attr=edge_attr)
        return graph
    
    def triangles_to_edges(self, cells):
        """Computes mesh edges from triangles."""
        # collect edges from triangles
        t_cell = torch.tensor(cells)
        edge_index = torch.cat((t_cell[:, :2], t_cell[:, 1:3], torch.cat((t_cell[:, 2].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1)), 0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single torch long tensor
        r, _ = torch.min(edge_index, 1, keepdim=True)
        s, _ = torch.max(edge_index, 1, keepdim=True)
        packed_edges = torch.cat((s, r), 1).type(torch.long)
        # remove duplicates and unpack
        unique_edges = torch.unique(packed_edges, dim=0)
        s, r = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0)))

    def training_step(self, batch, batch_idx):
        # assume batch size of 1 
        inputs = batch["x"]  # 1 t m c 
        pos = batch["pos"]
        pos = pos[:, 0, :, :2] # 1 m 2
        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        t = torch.randint(0, inputs.shape[1]-1, (1,)).item()
        x = inputs[:, t] # 1 m c
        y = inputs[:, t+1] # 1 m c

        # add extra time dimension
        pred = self(x.unsqueeze(1), pos) # 1 m c

        loss = F.mse_loss(pred, y) # 1 m c, 1 m c

        self.log("train/mse_loss", loss, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True,
                      sync_dist=self.dist,)
        
        labels_denorm = self.normalizer.denormalize(y) # 1 m c
        pred_denorm = self.normalizer.denormalize(pred) # 1 m c 

        train_loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        self.log("train/l1_loss", train_loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)

        return loss 

    def validation_step(self, batch, batch_idx, eval=False ):
        # assume batch size of 1 
        inputs = batch["x"]  # 1 t m c 
        pos = batch["pos"]
        pos = pos[:, 0, :, :2] # 1 m 2
        inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]

        t = torch.randint(0, inputs.shape[1]-1, (1,)).item()
        x = inputs[:, t] # 1 m c
        y = inputs[:, t+1] # 1 m c

        # add extra time dimension
        pred = self(x.unsqueeze(1), pos) # 1 m c
        loss = F.mse_loss(pred, y) # 1 m c, 1 m c

        labels_denorm = self.normalizer.denormalize(y) # 1 m c
        pred_denorm = self.normalizer.denormalize(pred) # 1 m c 

        loss_denorm = F.l1_loss(labels_denorm, pred_denorm) # single step error 
        
        rollout_error = 0 

        all_errors = []
        all_preds = torch.zeros_like(inputs) # 1 t m c
        pred_step = inputs[:, 0] # 1 m c
        all_preds[:, 0] = pred_step
        t = inputs.shape[1]

        for i in range(t-1):
            pred_step = self(pred_step.unsqueeze(1), pos) # 1 m c
            single_step_error = F.l1_loss(pred_step, inputs[:, i+1])
            all_errors.append(single_step_error.item())
            all_preds[:, i+1] = pred_step

        if eval:
            return all_errors, all_preds
        
        rollout_error = F.l1_loss(self.normalizer.denormalize(all_preds), self.normalizer.denormalize(inputs))

        self.log("val/mse_loss", loss, prog_bar=False,
                      logger=True, on_step=False, on_epoch=True,
                      sync_dist=self.dist,)
        self.log("val/l1_loss", loss_denorm, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)
        self.log("val/rollout_loss", rollout_error, prog_bar=False,
                        logger=True, on_step=False, on_epoch=True,
                        sync_dist=self.dist,)
        
        return loss

    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()),
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
