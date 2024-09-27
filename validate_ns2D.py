from dataset.datamodule import FluidsDataModule
from modules.models.baselines.ns2D import NS2DModule
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_grid
import pickle 
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time 

def main(args):
    config = get_yaml(args.config)
    device = args.device
    load_dir = config["load_dir"]
    dataconfig = config['data']
    verbose = args.verbose
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval/"
    os.makedirs(root_dir, exist_ok=True)

    modelconfig = config['model']
    trainconfig = config['training']
    path = config["model_path"]

    datamodule = FluidsDataModule(dataconfig)

    pl_module = NS2DModule(modelconfig=modelconfig,
                           trainconfig=trainconfig,
                           normalizer=datamodule.normalizer,
                           batch_size=dataconfig["batch_size"],)
    
    if path is not None:
        checkpoint = torch.load(path, map_location=device)
        pl_module.load_state_dict(checkpoint["state_dict"])
        print("Model loaded from: ", path)
    else:
        print("No model path given, using random weights")

    pl_module.eval()
    pl_module = pl_module.to(device)
    valid_loader = datamodule.val_dataloader()

    num_samples = 32*19

    plot_interval = 100
    all_losses = []
    all_times = []

    print("Number of samples: ", num_samples)
    print("Plot interval: ", plot_interval)

    idx = 0 
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch = {k: v.to(pl_module.device) for k, v in batch.items()}
            start = time.time()
            errors, rec = pl_module.validation_step(batch, 0, eval=True)
            end = time.time()
        
            x = batch["x"].detach().cpu() # b, t, x, y, c
            rec = rec.detach().cpu() # b, t, x, y, c

            if idx % plot_interval == 0:
                plot_grid(x[0, :, :, :, -1], rec[0, :, :, :, -1], n_t=5, path=root_dir + f"/plot_{idx}.png")

                plt.plot(errors)
                plt.savefig(root_dir + f"errors_{idx}.png")
                plt.close()

                save_dict = {"x": x, "rec": rec, "errors": errors}

                with open(root_dir + f"results_{idx}.pkl", "wb") as f:
                    pickle.dump(save_dict, f)

            rec_loss = F.l1_loss(x, rec)
            all_losses.append(rec_loss)
            all_times.append(end - start)
            idx += 1

            if verbose:
                print("Loss: ", rec_loss)
                print("Time: ", end - start)
    
    del all_times[0] # first time is always longer

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    with open(root_dir + "times.pkl", "wb") as f:
        pickle.dump(all_times, f)
    
    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an LDM')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", default=False)
    args = parser.parse_args()

    main(args)