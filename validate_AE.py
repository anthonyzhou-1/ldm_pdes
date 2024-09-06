from dataset.datamodule import FluidsDataModule
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_mesh, plot_grid
import pickle 
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse 

def validate_ns2D(config, device):
    aeconfig = config['model']['aeconfig']
    lossconfig = config['model']['lossconfig']
    trainconfig = config['training']
    dataconfig = config['data']

    root_dir = config['load_dir'] + "eval/"
    os.makedirs(root_dir, exist_ok=True)

    datamodule = FluidsDataModule(dataconfig)
    from modules.models.ae.ae_grid import Autoencoder, AutoencoderKL
    if "loss" in lossconfig.keys(): # use more complex autoencoder w/ GAN and LPIPS
        ae = Autoencoder
    else:
        ae = AutoencoderKL

    pl_module = ae(aeconfig, 
                    lossconfig, 
                    trainconfig,
                    normalizer=datamodule.normalizer,
                    batch_size=dataconfig["batch_size"],
                    accumulation_steps=trainconfig["accumulate_grad_batches"],)
    
    path = config["model_path"]
    checkpoint = torch.load(path, map_location=device)
    if "state_dict" in checkpoint.keys(): # sometimes the checkpoint is nested
        checkpoint = checkpoint["state_dict"]
    pl_module.load_state_dict(checkpoint)
    pl_module.eval()
    pl_module = pl_module.to(device)

    print("AE Model loaded from: ", path)
    valid_loader = datamodule.val_dataloader()

    num_samples = 32*19

    plot_interval = 10
    all_losses = []

    print("Number of samples: ", num_samples)
    print("Plot interval: ", plot_interval)

    idx = 0 
    print("Saving plots to: ", root_dir)
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch = {k: v.to(pl_module.device) for k, v in batch.items()}
            rec = pl_module.validation_step(batch, 0, eval=True)
        x = batch["x"].detach().cpu()    # b nt nx ny c
        rec = rec.detach().cpu()         # b nt nx ny c

        if idx % plot_interval == 0:
            u_batch = x[0, :, :, :, -1]      # nt nx ny, density
            rec_batch = rec[0, :, :, :, -1]  # nt nx ny
            
            path_u = root_dir + f"plot_{idx}.png"
            
            plot_grid(u_batch, rec_batch, n_t=5, path=path_u)

            save_dict = {"x": x, "rec": rec}

            with open(root_dir + f"results_{idx}.pkl", "wb") as f:
                pickle.dump(save_dict, f)
        loss = F.l1_loss(x, rec) 
        all_losses.append(loss)
        idx += 1

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    
    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))

def validate_cylinder(config, device):
    aeconfig = config['model']['aeconfig']
    lossconfig = config['model']['lossconfig']
    trainconfig = config['training']
    dataconfig = config['data']

    root_dir = config['load_dir'] + "eval/"
    os.makedirs(root_dir, exist_ok=True)

    datamodule = FluidsDataModule(dataconfig)
    normalizer = datamodule.normalizer
    from modules.models.ae.ae_mesh import AutoencoderKL, Autoencoder
    if "loss" in lossconfig.keys(): # use more complex autoencoder w/ GAN and LPIPS
        ae = Autoencoder
    else:
        ae = AutoencoderKL

    pl_module = ae(aeconfig, 
                    lossconfig, 
                    trainconfig,
                    normalizer=datamodule.normalizer,
                    batch_size=dataconfig["batch_size"],
                    accumulation_steps=trainconfig["accumulate_grad_batches"],)

    path = config["model_path"]
    checkpoint = torch.load(path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]

    pl_module.load_state_dict(checkpoint)
    pl_module.eval()
    pl_module = pl_module.to(device)

    print("AE Model loaded from: ", path)

    valid_loader = datamodule.val_dataloader()

    num_samples = len(valid_loader.dataset)

    plot_interval = 10
    all_losses = []

    print("Number of samples: ", num_samples)
    print("Plot interval: ", plot_interval)
    print("Saving plots to: ", root_dir)
    for idx in tqdm(range(0, num_samples)):
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

        if idx % plot_interval == 0:
            mesh_pos_batch = pos[0, 0, :, :2].detach().cpu() # m, 2
            cells_batch = cells # n_edges, 3
            u_batch = x[0, :, :, 0] # t, m
            rec_batch = rec[0, :, :, 0] # t, m
            
            path_u = root_dir + f"plot_{idx}.png"
            
            plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, rec=rec_batch, path=path_u)

        loss = F.l1_loss(x, rec) 
        all_losses.append(loss)

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))

def main(args):
    config=get_yaml(args.config)
    mode = config['data']['mode'] # get mode
    config["training"]['devices'] = 1 # set devices to 1
    device = args.device

    if mode == "cylinder":
        validate_cylinder(config, device)
    elif mode == "ns2D":
        validate_ns2D(config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an AE')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)