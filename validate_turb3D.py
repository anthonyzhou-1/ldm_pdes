from dataset.datamodule import FluidsDataModule
from modules.models.ddpm3D import LatentDiffusion
from modules.models.baselines.turb3D import Turb3DModule
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_3d_batch, plot_3d_rows
import pickle 
from einops import rearrange
import os
import os.path
from tqdm import tqdm
import argparse
import time 
from modules.losses.loss import ScaledLpLoss, LogTKESpectrumL2Distance, TurbulentKineticEnergySpectrum
import matplotlib.pyplot as plt

def validate_baseline(config, device):
    load_dir = config["load_dir"]
    dataconfig = config['data']
    batch_size = 1
    verbose = True 
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval/"
    os.makedirs(root_dir, exist_ok=True)

    modelconfig = config['model']
    trainconfig = config['training']
    path = config["model_path"]

    datamodule = FluidsDataModule(dataconfig)

    pl_module = Turb3DModule(modelconfig=modelconfig,
                       trainconfig=trainconfig,
                       normalizer=datamodule.normalizer,
                       batch_size=dataconfig["batch_size"],
                       accumulation_steps=trainconfig["accumulate_grad_batches"],)
    
    if path is not None:
        checkpoint = torch.load(path, map_location=device)
        pl_module.load_state_dict(checkpoint["state_dict"])
        print("Model loaded from: ", path)
    else:
        print("No model path given, using random weights")

    pl_module.eval()
    pl_module = pl_module.to(device)

    valid_loader = datamodule.val_dataloader()
    dataset = valid_loader.dataset 

    num_samples = len(valid_loader.dataset) # should be 45

    criterion2 = ScaledLpLoss(p=2)
    criterion1 = torch.nn.L1Loss()
    tke_criterion = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())
    tke_criterion = tke_criterion.to(device)

    plot_interval = 1
    all_losses1 = []
    all_losses2 = []
    all_times = []
    all_losses_tke = []

    with torch.no_grad():
        for idx in tqdm(range(0, num_samples)):
            if idx % plot_interval != 0:
                idx += 1
                continue
            batch = dataset.__getitem__(idx, eval=True)
            batch = {k: torch.tensor(v).unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 
            start = time.time()
            errors, rec = pl_module.validation_step(batch, 0, eval=True)
            end = time.time()

            x = batch["x"]   # b nt nz nx ny c
            rec = rec        # b nt nz nx ny c
            errors = errors.detach().cpu() # nt
            
            plot_3d_batch(x.detach().cpu(), t=[0.0, 0.3, 0.7, 1.0], path=root_dir + f"/inputs_{idx}.png")
            plot_3d_batch(rec.detach().cpu(), t=[0.0, 0.3, 0.7, 1.0], path=root_dir + f"/rec_{idx}.png")

            save_dict = {"x": x, "rec": rec}

            with open(root_dir + f"/log_{idx}.pkl", "wb") as f:
                pickle.dump(save_dict, f)

            plt.plot(errors)
            plt.savefig(root_dir + f"/errors_{idx}.png")
            plt.close()

            log = {"errors": errors, "rec": rec, "x": x}
            with open(root_dir + f"/log_{idx}.pkl", "wb") as f:
                pickle.dump(log, f)
            
            loss1 = criterion1(rec, x)
            loss2 = criterion2(rec, x)

            samples_tke = rearrange(rec, 'b t d h w c -> b c t d h w')
            inputs_tke = rearrange(x, 'b t d h w c -> b c t d h w')
            tke_losses = []
            for i in range(4):
                idx_start = i*24
                tke_loss, _, _, _ = tke_criterion(samples_tke[:, :, :, idx_start:idx_start+24], inputs_tke[:, :, :, idx_start:idx_start+24])
                tke_losses.append(tke_loss)
            mean_tke_loss = torch.mean(torch.stack(tke_losses))  
            all_losses1.append(loss1)
            all_losses2.append(loss2)
            all_losses_tke.append(mean_tke_loss)
            all_times.append(end - start)

            if verbose:
                print("Loss1: ", loss1)
                print("Loss2: ", loss2)
                print("TKE Loss: ", mean_tke_loss)
                print("Time: ", end - start)
                print("\n")

    del all_times[0] # remove first time as it is usually an outlier

    with open(root_dir + "losses1.pkl", "wb") as f:
        pickle.dump(all_losses1, f)
    
    with open(root_dir + "losses2.pkl", "wb") as f:
        pickle.dump(all_losses2, f)

    with open(root_dir + "losses_tke.pkl", "wb") as f:
        pickle.dump(all_losses_tke, f)

    with open(root_dir + "times.pkl", "wb") as f:
        pickle.dump(all_times, f)

    with open(root_dir + "mean_loss1.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses1))))
    
    with open(root_dir + "mean_loss2.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses2))))
    
    with open(root_dir + "mean_loss_tke.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses_tke))))

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses1)))
    print("Mean L2 Loss: ", torch.mean(torch.tensor(all_losses2)))
    print("Mean TKE Loss: ", torch.mean(torch.tensor(all_losses_tke)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))


def validate_turb3D(config, device):
    load_dir = config["load_dir"]
    dataconfig = config['data']
    verbose = True
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval_true/"
    os.makedirs(root_dir, exist_ok=True)

    datamodule = FluidsDataModule(dataconfig)

    pl_module = LatentDiffusion(**config["model"],
                                normalizer=datamodule.normalizer,
                                use_embed=dataconfig["dataset"]["use_embed"])

    path = config["model_path"]
    if path is not None:
        checkpoint = torch.load(path, map_location=device)
        if "state_dict" in checkpoint.keys(): # sometimes the checkpoint is nested
            checkpoint = checkpoint["state_dict"]
        pl_module.load_state_dict(checkpoint)
        print("LDM Model loaded from: ", path)
    else:
        print("No model path given, using random weights")
    pl_module.eval()
    pl_module = pl_module.to(device)

    valid_loader = datamodule.val_dataloader()
    dataset = valid_loader.dataset 

    num_samples = len(valid_loader.dataset) # should be 45

    criterion2 = ScaledLpLoss(p=2)
    criterion1 = torch.nn.L1Loss()
    tke_criterion = LogTKESpectrumL2Distance(TurbulentKineticEnergySpectrum())
    tke_criterion = tke_criterion.to(device)

    plot_interval = 1
    all_losses1 = []
    all_losses2 = []
    all_times = []
    all_losses_tke = []

    with torch.no_grad():
        for idx in tqdm(range(0, num_samples)):
            if idx % plot_interval != 0:
                idx += 1
                continue

            if os.path.isfile(root_dir + f"/log_{idx}.pkl"):
                print("Skipping idx: ", idx)
                log = pickle.load(open(root_dir + f"/log_{idx}.pkl", "rb"))
                start = 0
                end = 0
            else:
                batch = dataset.__getitem__(idx, eval=True)

                use_prompt = False 
                if 'prompt' in batch.keys():
                    prompt = batch.pop("prompt")
                    with open(root_dir + f"/prompt_{idx}.txt", "w") as text_file:
                        text_file.write(prompt)
                    use_prompt = True

                batch = {k: torch.tensor(v).unsqueeze(0).to(pl_module.device) for k, v in batch.items()} 
                
                if use_prompt:
                    batch["prompt"] = [prompt]

                start = time.time()
                log = pl_module.log_images(batch)
                end = time.time()

                with open(root_dir + f"/log_{idx}.pkl", "wb") as f:
                    pickle.dump(log, f)

                # inputs, rec, samples in shape b t d h w c
                # diffusion_row, denoise_row in shape n_steps b t d h w c
                t = [0, 0.3, 0.7, 1.0]
                inputs = log["inputs"].detach().cpu() # b t d h w c
                path_inputs = root_dir + f"/inputs_{idx}.png" 
                plot_3d_batch(inputs, t, path=path_inputs)

                reconstruction = log["reconstruction"].detach().cpu() # b t d h w c
                path_rec = root_dir + f"/reconstruction_{idx}.png"
                plot_3d_batch(reconstruction, t, path=path_rec)

                samples = log["samples"].detach().cpu() # b t d h w c
                path_samples = root_dir + f"/samples_{idx}.png"
                plot_3d_batch(samples, t, path=path_samples)

                diffusion_row = log["diffusion_row"].detach().cpu() # n_steps b t d h w c
                path_diffusion = root_dir + f"/forward_diffusion_{idx}.png"
                plot_3d_rows(diffusion_row, t, path=path_diffusion)

                denoise_row = log["denoise_row"].detach().cpu() # n_steps b t d h w c
                path_denoise = root_dir + f"/reverse_diffusion_{idx}.png"
                plot_3d_rows(denoise_row, t, path=path_denoise)

            loss1 = criterion1(log["samples"], log["inputs"])
            loss2 = criterion2(log["samples"], log["inputs"])

            samples_tke = rearrange(log['samples'], 'b t d h w c -> b c t d h w')
            inputs_tke = rearrange(log['inputs'], 'b t d h w c -> b c t d h w')
            tke_losses = []
            for i in range(4):
                idx_start = i*24
                tke_loss, _, _, _ = tke_criterion(samples_tke[:, :, :, idx_start:idx_start+24], inputs_tke[:, :, :, idx_start:idx_start+24])
                tke_losses.append(tke_loss)
            mean_tke_loss = torch.mean(torch.stack(tke_losses))  
            all_losses1.append(loss1)
            all_losses2.append(loss2)
            all_losses_tke.append(mean_tke_loss)
            all_times.append(end - start)

            if verbose:
                print("Loss1: ", loss1)
                print("Loss2: ", loss2)
                print("TKE Loss: ", mean_tke_loss)
                print("Time: ", end - start)
                print("\n")

    del all_times[0] # remove first time as it is usually an outlier

    with open(root_dir + "losses1.pkl", "wb") as f:
        pickle.dump(all_losses1, f)
    
    with open(root_dir + "losses2.pkl", "wb") as f:
        pickle.dump(all_losses2, f)

    with open(root_dir + "losses_tke.pkl", "wb") as f:
        pickle.dump(all_losses_tke, f)

    with open(root_dir + "times.pkl", "wb") as f:
        pickle.dump(all_times, f)

    with open(root_dir + "mean_loss1.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses1))))
    
    with open(root_dir + "mean_loss2.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses2))))
    
    with open(root_dir + "mean_loss_tke.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses_tke))))

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses1)))
    print("Mean L2 Loss: ", torch.mean(torch.tensor(all_losses2)))
    print("Mean TKE Loss: ", torch.mean(torch.tensor(all_losses_tke)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))

def main(args):
    config=get_yaml(args.config)
    config['verbose'] = args.verbose
    mode = config['data']['mode'] # get mode
    config["training"]['devices'] = 1 # set devices to 1
    device = args.device

    if "name" in config['model'].keys():
        validate_baseline(config, device)
    elif mode == "turb3D":
        validate_turb3D(config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an LDM')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", default=False) 
    args = parser.parse_args()

    main(args)