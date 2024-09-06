from dataset.datamodule import FluidsDataModule
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_mesh
import pickle 
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def validate_cylinder(config, device):
    dataconfig = config['data']
    modelconfig = config['model']
    trainconfig = config['training']

    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = config['load_dir'] + "validation"
    os.makedirs(root_dir, exist_ok=True)
    
    datamodule = FluidsDataModule(dataconfig)

    model_name = config['model_name']

    if model_name == "oformer":
        from modules.models.baselines.oformer.oformer import OFormerModule
        pl_module = OFormerModule(modelconfig=modelconfig,
                       trainconfig=trainconfig,
                       normalizer=datamodule.normalizer,
                       batch_size=dataconfig["batch_size"],
                       accumulation_steps=trainconfig["accumulate_grad_batches"],)
    elif model_name == "gnn":
        from modules.models.baselines.gnn import GNNModule
        pl_module = GNNModule(modelconfig=modelconfig,
                       trainconfig=trainconfig,
                       normalizer=datamodule.normalizer,
                       batch_size=dataconfig["batch_size"],
                       accumulation_steps=trainconfig["accumulate_grad_batches"],)
    elif model_name == "gino":
        from modules.models.baselines.gino import GINOModule
        pl_module = GINOModule(modelconfig=modelconfig['gino'],
                          trainconfig=trainconfig,
                          latent_grid_size=modelconfig["latent_grid_size"],
                          normalizer=datamodule.normalizer,
                          batch_size=dataconfig["batch_size"],
                          accumulation_steps=trainconfig["accumulate_grad_batches"],)

    path = config["model_path"]
    checkpoint = torch.load(path, map_location=device)
    pl_module.load_state_dict(checkpoint["state_dict"])
    pl_module.eval()
    pl_module = pl_module.to(device)

    print("Model loaded from: ", path)

    valid_loader = datamodule.val_dataloader()
    normalizer = datamodule.normalizer

    plot_interval = 10
    all_losses = []

    idx = 0 
    for batch in tqdm(valid_loader):
        if idx % plot_interval != 0:
            idx += 1
            continue

        batch = {k: v.to(pl_module.device) for k, v in batch.items()}

        with torch.no_grad():
            errors, rec = pl_module.validation_step(batch, 0, eval=True)
    
        x = batch["x"].detach().cpu() # b, t, m, c
        pos = batch["pos"].detach().cpu() # b, t, m, 3
        rec = rec.detach().cpu() # b t m c 

        # x remains in unnormalized form, but rec needs to be denormalized
        rec = normalizer.denormalize(rec)

        x = x.squeeze()
        rec = rec.squeeze()

        if idx % plot_interval == 0:
            mesh_pos_batch = pos[0, 0, :, :2]
            cells = batch["cells"]
            cells_batch = cells.squeeze().detach().cpu() # n_edges, 3
            u_batch = x[:, :, 0] # t, m
            rec_batch = rec[:, :, 0] # t, m
            
            path_u = root_dir + f"result_{idx}.png"
            
            plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u, rec=rec_batch)

            plt.plot(errors)
            plt.savefig(root_dir + f"errors_{idx}.png")
            plt.close()

            save_dict = {"x": x, "rec": rec, "mesh_pos": mesh_pos_batch, "cells": cells_batch, "errors": errors}

            with open(root_dir + f"results_{idx}.pkl", "wb") as f:
                pickle.dump(save_dict, f)

        rec_loss = F.l1_loss(x, rec)
        all_losses.append(rec_loss)
        idx += 1
    
    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    
    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))


def main(args):
    config = get_yaml(args.config)
    validate_cylinder(config, args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an LDM')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)