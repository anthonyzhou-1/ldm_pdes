from dataset.datamodule import FluidsDataModule
from modules.models.ae.ae_mesh import Autoencoder, AutoencoderKL
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_mesh
import pickle 
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse 

def validate_AE(config, device):
    config=get_yaml("./configs/unstructured/ae_gino_16x16x16x16_normed_batched.yaml")

    aeconfig = config['model']['aeconfig']
    lossconfig = config['model']['lossconfig']
    trainconfig = config['training']
    dataconfig = config['data']

    root_dir = f"./logs/eval/val_ae/"
    os.makedirs(root_dir, exist_ok=True)

    datamodule = FluidsDataModule(**dataconfig)
    normalizer = datamodule.normalizer

    pl_module = AutoencoderKL(aeconfig, 
                            lossconfig, 
                            trainconfig,
                            normalizer=datamodule.normalizer,
                            batch_size=dataconfig["batch_size"],
                            accumulation_steps=trainconfig["accumulate_grad_batches"],
                            padding=dataconfig["padding"],)

    path = "logs/GINO_normal_16/rec_loss=0.01.ckpt"
    checkpoint = torch.load(path, map_location=device)
    pl_module.load_state_dict(checkpoint["state_dict"])
    pl_module.eval()
    pl_module = pl_module.to(device)

    print("AE Model loaded from: ", path)

    valid_loader = datamodule.val_dataloader()

    num_samples = len(valid_loader.dataset)

    plot_interval = 20
    all_losses = []

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
            
            path_u = root_dir + f"/u_{idx}.png"
            path_rec = root_dir + f"/rec_{idx}.png"
            
            plot_mesh(u_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_u)
            plot_mesh(rec_batch, mesh_pos_batch, cells_batch, n_t=5, path=path_rec)

        loss = F.l1_loss(x, rec) 
        all_losses.append(loss)

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))

def main(args):
    config = get_yaml(args.config)
    validate_AE(config, args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an AE')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)