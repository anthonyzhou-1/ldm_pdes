from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch 
import torch.nn.functional as F
import wandb
import pickle 
from modules.modules.plotting import plot_mesh, plot_mesh_batch, plot_grid, plot_grid_batch, plot_3d, plot_3d_batch, plot_3d_rows
import matplotlib.pyplot as plt 
import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable
from modules.losses.loss import ScaledLpLoss, LogTKESpectrumL2Distance, TurbulentKineticEnergySpectrum
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class ACDMCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        #if pl_module.current_epoch == 0:
        #    return 
        
        if pl_module.global_rank == 0:
            valid_loader = trainer.val_dataloaders
            dataset = valid_loader.dataset 
            batch = next(dataset.__iter__())

            batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 

            with torch.no_grad():
                log = pl_module.log_images(batch)

            with open(trainer.default_root_dir + "/log.pkl", "wb") as f:
                pickle.dump(log, f)

            # true_rollout, full_rollout in shape b t x y c

            inputs = log["true_rollout"][0, :, :, :, -1].detach().cpu() # t x y
            path_inputs = trainer.default_root_dir + "/inputs.png" 
            plot_grid(inputs, n_t=4, path=path_inputs)

            samples = log["full_rollout"][0, :, :, :, -1].detach().cpu() # t, x y
            path_samples = trainer.default_root_dir + "/samples.png"
            plot_grid(samples, n_t=4, path=path_samples)

            rec_loss = F.l1_loss(log["true_rollout"], log["full_rollout"])
            wandb.log({"val/rec_loss": rec_loss, "trainer/global_step": trainer.global_step})

class OFormerCallback(Callback):
    #def on_validation_epoch_end(self, trainer, pl_module):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        valid_loader = trainer.val_dataloaders
        normalizer = pl_module.normalizer

        idx = 10

        batch = valid_loader.dataset.__getitem__(idx, eval=True)
        cells = batch.pop("cells")

        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}

        with torch.no_grad():
            errors, rec = pl_module.validation_step(batch, 0, eval=True)

        x = batch["x"].detach().cpu() # b, t, m, c
        pos = batch["pos"].detach().cpu() # b, t, m, 3
        rec = rec.detach().cpu() # b t m c 

        # x remains in unnormalized form, but rec needs to be denormalized
        rec = normalizer.denormalize(rec)

        mesh_pos_batch = pos[0, 0, :, :2] # m, 2
        cells_batch = cells.squeeze().detach().cpu() # n_edges, 3
        u_batch = x[0, :, :, 0] # t, m
        rec_batch = rec[0, :, :, 0] # t, m
        
        path_u = trainer.default_root_dir + "/u.png"
        path_rec = trainer.default_root_dir + "/pred.png"
        
        plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u)
        plot_mesh(rec_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_rec)

        plt.plot(errors)
        plt.savefig(trainer.default_root_dir + "/errors.png")
        plt.close()

        save_dict = {"x": x, "rec": rec, "mesh_pos": mesh_pos_batch, "cells": cells_batch, "errors": errors}

        with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)


class GNNCallback(Callback):
    #def on_validation_epoch_end(self, trainer, pl_module):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        valid_loader = trainer.val_dataloaders
        normalizer = pl_module.normalizer

        idx = 10

        batch = valid_loader.dataset.__getitem__(idx)

        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}

        with torch.no_grad():
            errors, rec = pl_module.validation_step(batch, 0, eval=True)

        x = batch["x"].detach().cpu() # b, t, m, c
        pos = batch["pos"].detach().cpu() # b, t, m, 3
        rec = rec.detach().cpu() # t m c 

        # x remains in unnormalized form, but rec needs to be denormalized
        rec = normalizer.denormalize(rec)

        mesh_pos_batch = pos[0, 0, :, :2] # m, 2
        cells_batch = batch['cells'].squeeze().detach().cpu() # n_edges, 3
        u_batch = x[0, :, :, 0] # t, m
        rec_batch = rec[:, :, 0] # t, m
        
        path_u = trainer.default_root_dir + "/u.png"
        path_rec = trainer.default_root_dir + "/pred.png"
        
        plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u)
        plot_mesh(rec_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_rec)

        plt.plot(errors)
        plt.savefig(trainer.default_root_dir + "/errors.png")
        plt.close()

        save_dict = {"x": x, "rec": rec, "mesh_pos": mesh_pos_batch, "cells": cells_batch, "errors": errors}

        with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)


class NS2DCallback(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
    #def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        valid_loader = trainer.val_dataloaders
        dataset = valid_loader.dataset 
        batch = next(dataset.__iter__())

        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 
        
        with torch.no_grad():
            errors, rec = pl_module.validation_step(batch, 0, eval=True)
        
        x = batch["x"].detach().cpu() # b, t, x, y, c
        rec = rec.detach().cpu() # b, t, x, y, c

        plot_grid(x[0, :, :, :, -1], rec[0, :, :, :, -1], n_t=4, path=trainer.default_root_dir + "/plot.png")

        plt.plot(errors)
        plt.savefig(trainer.default_root_dir + "/errors.png")
        plt.close()

        save_dict = {"x": x, "rec": rec, "errors": errors}

        with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)

class GINOCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        valid_loader = trainer.val_dataloaders
        normalizer = pl_module.normalizer

        idx = 10

        batch = valid_loader.dataset.__getitem__(idx, eval=True)

        cells = batch.pop("cells")
        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}

        with torch.no_grad():
            errors, rec = pl_module.validation_step(batch, 0, eval=True)

        x = batch["x"].detach().cpu() # b, t, m, c
        pos = batch["pos"].detach().cpu() # b, t, m, 3
        rec = rec.detach().cpu()

        # x remains in unnormalized form, but rec needs to be denormalized
        rec = normalizer.denormalize(rec)

        mesh_pos_batch = pos[0, 0, :, :2] # m, 2
        cells_batch = cells # n_edges, 3
        u_batch = x[0, :, :, 0] # t, m
        rec_batch = rec[0, :, :, 0] # t, m
        
        path_u = trainer.default_root_dir + "/u.png"
        path_rec = trainer.default_root_dir + "/rec.png"
        
        plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u)
        plot_mesh(rec_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_rec)

        plt.plot(errors)
        plt.savefig(trainer.default_root_dir + "/errors.png")
        plt.close()

        save_dict = {"x": x, "rec": rec, "mesh_pos": mesh_pos_batch, "cells": cells_batch, "errors": errors}

        with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)

class Turb3DLDMCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_loss = 100
        self.l2 = ScaledLpLoss()
        self.tke = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())

    def on_validation_epoch_end(self, trainer, pl_module):
        #if pl_module.current_epoch == 0:
        #    return 
        
        if pl_module.global_rank == 0:
            try:
                with torch.no_grad():
                    valid_loader = trainer.val_dataloaders
                    dataset = valid_loader.dataset 
                    batch = dataset.__getitem__(0, eval=True)

                    use_prompt = False 
                    if 'prompt' in batch.keys():
                        prompt = batch.pop("prompt")
                        with open(trainer.default_root_dir + "/prompt.txt", "w") as text_file:
                            text_file.write(prompt)
                        use_prompt = True

                    batch = {k: torch.tensor(v).unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 
                    
                    if use_prompt:
                        batch["prompt"] = [prompt]
                    
                    log = pl_module.log_images(batch)

                    with open(trainer.default_root_dir + "/log.pkl", "wb") as f:
                        pickle.dump(log, f)

                    # inputs, rec, samples in shape b t d h w c
                    # diffusion_row, denoise_row in shape n_steps b t d h w c
                    t = [0, 0.3, 0.7, 1.0]
                    inputs = log["inputs"].detach().cpu() # b t d h w c
                    path_inputs = trainer.default_root_dir + "/inputs.png" 
                    plot_3d_batch(inputs, t, path=path_inputs)

                    reconstruction = log["reconstruction"].detach().cpu() # b t d h w c
                    path_rec = trainer.default_root_dir + "/reconstruction.png"
                    plot_3d_batch(reconstruction, t, path=path_rec)

                    samples = log["samples"].detach().cpu() # b t d h w c
                    path_samples = trainer.default_root_dir + "/samples.png"
                    plot_3d_batch(samples, t, path=path_samples)

                    diffusion_row = log["diffusion_row"].detach().cpu() # n_steps b t d h w c
                    path_diffusion = trainer.default_root_dir + "/forward_diffusion.png"
                    plot_3d_rows(diffusion_row, t, path=path_diffusion)

                    denoise_row = log["denoise_row"].detach().cpu() # n_steps b t d h w c
                    path_denoise = trainer.default_root_dir + f"/reverse_diffusion.png"
                    plot_3d_rows(denoise_row, t, path=path_denoise)

                    rec_loss = F.l1_loss(log["samples"], log["inputs"])
                    wandb.log({"val/rec_loss": rec_loss, "trainer/global_step": trainer.global_step})

                    l2_loss = self.l2(log["samples"], log["inputs"])
                    wandb.log({"val/rel_l2_loss": l2_loss, "trainer/global_step": trainer.global_step})

                    samples_tke = rearrange(log['samples'], 'b t d h w c -> b c t d h w')
                    inputs_tke = rearrange(log['inputs'], 'b t d h w c -> b c t d h w')
                    tke_losses = []
                    self.tke = self.tke.to(pl_module.device)
                    for i in range(4):
                        idx_start = i*24
                        tke_loss, _, _, _ = self.tke(samples_tke[:, :, :, idx_start:idx_start+24], inputs_tke[:, :, :, idx_start:idx_start+24])
                        tke_losses.append(tke_loss)
                    mean_tke_loss = torch.mean(torch.stack(tke_losses))
                    wandb.log({"val/tke_loss": mean_tke_loss, "trainer/global_step": trainer.global_step})

                    if l2_loss < self.best_loss:
                        self.best_loss = rec_loss
                        torch.save(pl_module.state_dict(), trainer.default_root_dir + "/best.ckpt")
                        with open(trainer.default_root_dir + "/best_loss.txt", "w") as text_file:
                            text_file.write(str(self.best_loss))
            except:
                print("Error in plotting callback")

class GridLDMCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_loss = 100

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            return 
        
        if pl_module.global_rank == 0:
            valid_loader = trainer.val_dataloaders
            dataset = valid_loader.dataset 
            batch = next(dataset.__iter__())

            if pl_module.use_embed: 
                prompt = batch.pop("prompt")
                with open(trainer.default_root_dir + "/prompt.txt", "w") as text_file:
                    text_file.write(prompt)

            batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 
            
            if pl_module.use_embed:
                batch["prompt"] = [prompt]

            with torch.no_grad():
                log = pl_module.log_images(batch)

            with open(trainer.default_root_dir + "/log.pkl", "wb") as f:
                pickle.dump(log, f)

            # inputs, rec, samples in shape b t x y c
            # diffusion_row, denoise_row in shape n_steps b t x y c

            inputs = log["inputs"][0, :, :, :, -1].detach().cpu() # t x y
            path_inputs = trainer.default_root_dir + "/inputs.png" 
            plot_grid(inputs, n_t=4, path=path_inputs)

            reconstruction = log["reconstruction"][0, :, :, :, -1].detach().cpu() # t, x y
            path_rec = trainer.default_root_dir + "/reconstruction.png"
            plot_grid(reconstruction, n_t=4, path=path_rec)

            samples = log["samples"][0, :, :, :, -1].detach().cpu() # t, x y
            path_samples = trainer.default_root_dir + "/samples.png"
            plot_grid(samples, n_t=4, path=path_samples)

            diffusion_row = log["diffusion_row"][:, 0, :, :, :, -1].detach().cpu() # n_steps, t, x y
            path_diffusion = trainer.default_root_dir + "/forward_diffusion.png"
            plot_grid_batch(diffusion_row, n_t=4, path=path_diffusion)

            denoise_row = log["denoise_row"][:, 0, :, :, :, -1].detach().cpu() # n_steps, t, x y
            path_denoise = trainer.default_root_dir + "/reverse_diffusion.png"
            plot_grid_batch(denoise_row, n_t=4, path=path_denoise)

            rec_loss = F.l1_loss(log["inputs"], log["samples"])
            wandb.log({"val/rec_loss": rec_loss, "trainer/global_step": trainer.global_step})

            if rec_loss < self.best_loss:
                self.best_loss = rec_loss
                torch.save(pl_module.state_dict(), trainer.default_root_dir + "/best.ckpt")
                with open(trainer.default_root_dir + "/best_loss.txt", "w") as text_file:
                    text_file.write(str(self.best_loss))

class MeshLDMCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_loss = 100

    def on_validation_epoch_end(self, trainer, pl_module):
        #if pl_module.current_epoch == 0:
        #    return 
        
        if pl_module.global_rank == 0:
            valid_loader = trainer.val_dataloaders
            batch = valid_loader.dataset.__getitem__(0, eval=True)

            cells = batch.pop("cells")

            use_prompt = True if 'prompt' in batch.keys() else False

            if use_prompt:
                prompt = batch.pop("prompt")
                with open(trainer.default_root_dir + "/prompt.txt", "w") as text_file:
                    text_file.write(prompt)

            batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}
            if use_prompt:
                batch["prompt"] = [prompt]

            with torch.no_grad():
                log = pl_module.log_images(batch)

            with open(trainer.default_root_dir + "/log.pkl", "wb") as f:
                log['cells'] = cells    
                if pl_module.use_embed:
                    log['prompt'] = prompt
                pickle.dump(log, f)

            # inputs, rec, samples in shape b t m c
            # diffusion_row, denoise_row in shape n_steps, b, t, m, c
            pos = batch["pos"] # b, t, m, 3

            if "pad_mask" in batch.keys(): # need to trim padded data
                length = torch.sum(batch["pad_mask"][0], dtype=torch.long)
                log['inputs'] = log['inputs'][:, :, :length] # 1 t m c
                log['samples'] = log['samples'][:, :, :length] # 1 t m c
                log['reconstruction'] = log['reconstruction'][:, :, :length] # 1 t m c
                log['diffusion_row'] = log['diffusion_row'][:, :, :, :length] # n_steps b t m c
                log['denoise_row'] = log['denoise_row'][:, :, :, :length] # n_steps b t m c
                pos = pos[:, :, :length] # b t m 3

            mesh_pos = pos[0, 0, :, :2].detach().cpu() # m, 2
            inputs = log["inputs"][0, :, :, 0].detach().cpu() # t, m
            path_inputs = trainer.default_root_dir + "/inputs.png" 
            plot_mesh(inputs, mesh_pos, cells, n_t=4, path=path_inputs)

            reconstruction = log["reconstruction"][0, :, :, 0].detach().cpu() # t, m
            path_rec = trainer.default_root_dir + "/reconstruction.png"
            plot_mesh(reconstruction, mesh_pos, cells, n_t=4, path=path_rec)

            samples = log["samples"][0, :, :, 0].detach().cpu() # t, m
            path_samples = trainer.default_root_dir + "/samples.png"
            plot_mesh(samples, mesh_pos, cells, n_t=4, path=path_samples)

            diffusion_row = log["diffusion_row"][:, 0, :, :, 0].detach().cpu() # n_steps, t, m
            path_diffusion = trainer.default_root_dir + "/forward_diffusion.png"
            plot_mesh_batch(diffusion_row, mesh_pos, cells, n_t=4, path=path_diffusion)

            denoise_row = log["denoise_row"][:, 0, :, :, 0].detach().cpu() # n_steps, t, m
            path_denoise = trainer.default_root_dir + "/reverse_diffusion.png"
            plot_mesh_batch(denoise_row, mesh_pos, cells, n_t=4, path=path_denoise)

            rec_loss = F.l1_loss(log["inputs"], log["samples"])
            wandb.log({"val/rec_loss": rec_loss, "trainer/global_step": trainer.global_step})

            if rec_loss < self.best_loss:
                self.best_loss = rec_loss
                torch.save(pl_module.state_dict(), trainer.default_root_dir + "/best.ckpt")
                with open(trainer.default_root_dir + "/best_loss.txt", "w") as text_file:
                    text_file.write(str(self.best_loss))

class MeshPlottingCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        
        if pl_module.global_rank == 0:
            try: 
                idx = 10

                valid_loader = trainer.val_dataloaders
                normalizer = pl_module.normalizer
                batch = valid_loader.dataset.__getitem__(idx, eval=True)

                cells = batch.pop("cells")

                batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}

                with torch.no_grad():
                    rec = pl_module.validation_step(batch, 0, eval=True)

                x = batch["x"].detach().cpu()    
                pos = batch["pos"].detach().cpu()
                rec = rec.detach().cpu()
                pad_mask = batch.get("pad_mask", None)

                if pad_mask is not None:
                    pad_mask = pad_mask.detach().cpu()
                    length = torch.sum(pad_mask[0], dtype=torch.long) 
                    x = x[:, :, :length]
                    rec = rec[:, :, :length]
                    pos = pos[:, :, :length]

                # x remains in unnormalized form, but rec needs to be denormalized
                rec = normalizer.denormalize(rec)

                mesh_pos_batch = pos[0, 0, :, :2].detach().cpu() # m, 2
                cells_batch = cells # n_edges, 3
                u_batch = x[0, :, :, 0] # t, m
                rec_batch = rec[0, :, :, 0] # t, m
                
                path_u = trainer.default_root_dir + "/u.png"
                path_rec = trainer.default_root_dir + "/rec.png"
                
                plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u)
                plot_mesh(rec_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_rec)

                save_dict = {"u": u_batch, "rec": rec_batch, "mesh_pos": mesh_pos_batch, "cells": cells_batch}

                with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
                    pickle.dump(save_dict, f)
            except:
                print("Error in plotting callback")

class GridPlottingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.global_rank == 0:
            try: 
                valid_loader = trainer.val_dataloaders
                dataset = valid_loader.dataset 
                batch = next(dataset.__iter__())
                batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 

                with torch.no_grad():
                    rec = pl_module.validation_step(batch, 0, eval=True)

                x = batch["x"].detach().cpu()    # b nt nx ny c
                rec = rec.detach().cpu()         # b nt nx ny c
                
                u_batch = x[0, :, :, :, -1]      # nt nx ny, density
                rec_batch = rec[0, :, :, :, -1]  # nt nx ny
                
                path_u = trainer.default_root_dir + "/plot.png"
                
                plot_grid(u_batch, rec_batch, n_t=5, path=path_u)

                save_dict = {"x": x, "rec": rec}

                with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
                    pickle.dump(save_dict, f)
            except:
                print("Error in plotting callback")

class BaselineCallback3D(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        valid_loader = trainer.val_dataloaders
        dataset = valid_loader.dataset 
        batch = dataset.__getitem__(0, eval=True)
        batch = {k: torch.tensor(v).unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 

        with torch.no_grad():
            all_errors, rec = pl_module.validation_step(batch, 0, eval=True)

        x = batch["x"].detach().cpu()    # b nt nz nx ny c
        rec = rec.detach().cpu()         # b nt nz nx ny c
        all_errors = all_errors.detach().cpu() # nt
        
        u_batch = x[0, :, :, :, :, :3]  # nt nz nx ny 3
        rec_batch = rec[0, :, :, :, :, :3]  # nt nz nx ny 3

        u_batch = torch.norm(u_batch, dim=-1) # nt nz nx ny
        rec_batch = torch.norm(rec_batch, dim=-1) # nt nz nx ny
        
        for t in [0.0, 0.5, 1.0]:
            path_u = trainer.default_root_dir + f"/u_{t}.png"
            path_rec = trainer.default_root_dir + f"/rec_{t}.png"

            plot_3d(u_batch, path=path_u, t=t)
            plot_3d(rec_batch, path=path_rec, t=t)

        save_dict = {"x": x, "rec": rec}

        with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)

        plt.plot(all_errors)
        plt.savefig(trainer.default_root_dir + "/errors.png")
        plt.close()

class PlottingCallback3D(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        #if pl_module.current_epoch == 0:
        #    return 
        if pl_module.global_rank == 0:
            try: 
                valid_loader = trainer.val_dataloaders
                dataset = valid_loader.dataset 
                batch = dataset.__getitem__(0, eval=True)
                batch = {k: torch.tensor(v).unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 

                with torch.no_grad():
                    rec = pl_module.validation_step(batch, 0, eval=True)

                x = batch["x"].detach().cpu()    # b nt nz nx ny c
                rec = rec.detach().cpu()         # b nt nz nx ny c
                
                u_batch = x[0, :, :, :, :, :3]  # nt nz nx ny 3 take only velocity components
                rec_batch = rec[0, :, :, :, :, :3]  # nt nz nx ny 3

                u_batch = torch.norm(u_batch, dim=-1) # nt nz nx ny
                rec_batch = torch.norm(rec_batch, dim=-1) # nt nz nx ny

                u_batch = torch.flip(u_batch, dims=[1, 3]) # orient in 3D space correctly just for plotting
                u_batch = torch.rot90(u_batch, k=1, dims=[2, 3])

                rec_batch = torch.flip(rec_batch, dims=[1, 3])
                rec_batch = torch.rot90(rec_batch, k=1, dims=[2, 3])
                
                for t in [0.0, 0.5, 1.0]:
                    path_u = trainer.default_root_dir + f"/u_{t}.png"
                    path_rec = trainer.default_root_dir + f"/rec_{t}.png"

                    plot_3d(u_batch, path=path_u, t=t)
                    plot_3d(rec_batch, path=path_rec, t=t)

                save_dict = {"x": x, "rec": rec}

                with open(trainer.default_root_dir + "/results.pkl", "wb") as f:
                    pickle.dump(save_dict, f)

            except:
                print("Error in plotting callback")

class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.

    https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self, decay: float, validate_original_weights: bool = False, every_n_steps: int = 1, cpu_offload: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device if not self.cpu_offload else torch.device('cpu')
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer): # swap to ema weights for validation
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer): # swap back to training weights for training 
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        # use the connector as NeMo calls the connector directly in the exp_manager when restoring.
        connector = trainer._checkpoint_connector
        # Replace connector._ckpt_path with below to avoid calling into lightning's protected API
        ckpt_path = trainer.ckpt_path

        if ckpt_path and checkpoint_callback is not None and 'NeMo' in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f'-EMA{ext}'):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f'-EMA{ext}')
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device('cpu'))

                checkpoint['optimizer_states'] = ema_state_dict['optimizer_states']
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group['params'])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        if getattr(self.optimizer, "_step_supports_amp_scaling", False) and grad_scaler is not None:
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )

            if self.device.type == 'cuda':
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu, args=(self.ema_params, current_model_state, self.decay, self.stream,),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(self.all_parameters())
        state_dict = {
            'opt': self.optimizer.state_dict(),
            'ema': ema_params,
            'current_step': self.current_step,
            'decay': self.decay,
            'every_n_steps': self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        self.optimizer.load_state_dict(state_dict['opt'])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict['ema']))
        self.current_step = state_dict['current_step']
        self.decay = state_dict['decay']
        self.every_n_steps = state_dict['every_n_steps']
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True