from dataset.datamodule import FluidsDataModule
from modules.models.ddpm import LatentDiffusion
import torch 
from modules.utils import get_yaml
from modules.modules.plotting import plot_mesh, plot_grid
import pickle 
import torch.nn.functional as F
import os
import os.path
from tqdm import tqdm
import argparse
import copy 
from modules.modules.phiflow import simulate_fluid, simulate_fluid_lowres
import time
from modules.losses.loss import ScaledLpLoss
from modules.modules.ddim import DDIMSampler

def validate_cylinder_ddim(config, device, ddim_steps, dir_name="eval"):
    load_dir = config["load_dir"]
    dataconfig = config['data']
    verbose = config['verbose']
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + f"{dir_name}/"
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

    num_samples = len(valid_loader.dataset) # should be 100 

    criterion_l2 = ScaledLpLoss(p=2)
    criterion_l1 = torch.nn.L1Loss()

    plot_interval = 1
    all_losses = []
    all_losses_l1 = []
    all_times = []

    sampler = DDIMSampler(model=pl_module)

    for idx in tqdm(range(0, num_samples)):
        if idx % plot_interval != 0:
            idx += 1
            continue
        batch = valid_loader.dataset.__getitem__(idx, eval=True)

        cells = batch.pop("cells")
        pad_mask = batch.get('pad_mask', None)
        cond = batch.get('cond', None)

        if pl_module.use_embed:
            prompt = batch.pop("prompt")

        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}
        if pl_module.use_embed:
            batch["prompt"] = [prompt]

        log = {}
        start = time.time()
        with torch.no_grad():
            z, c, x, xrec, xc = pl_module.get_input(batch, 
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=1)
            log["inputs"] = pl_module.normalizer.denormalize(x) 
            shape = (batch_size, pl_module.channels, pl_module.image_size[0], pl_module.image_size[1], pl_module.image_size[2])
            samples, _= sampler.sample(S=ddim_steps,
                                       batch_size=1,
                                       conditioning=c,
                                       shape=shape,
                                       verbose=False)
            x_samples = pl_module.decode_first_stage(samples, batch['pos'], pad_mask=pad_mask, cond=cond) # decode denoised latent into x_sample
            log["samples"] = x_samples

        end = time.time()
        pos = batch["pos"] # b, t, m, 3

        if "pad_mask" in batch.keys(): # need to trim padded data
            length = torch.sum(batch["pad_mask"][0], dtype=torch.long)
            log['inputs'] = log['inputs'][:, :, :length] # 1 t m c
            log['samples'] = log['samples'][:, :, :length] # 1 t m c
            pos = pos[:, :, :length] # b t m 3
        
        if idx % plot_interval == 0:

            mesh_pos = pos[0, 0, :, :2].detach().cpu() # m, 2

            inputs = log["inputs"][0, :, :, 0].detach().cpu() # t, m
            path_inputs = root_dir + f"inputs_{idx}.png" 
            plot_mesh(inputs, mesh_pos, cells, n_t=5, path=path_inputs)

            sample = log["samples"][0, :, :, 0].detach().cpu() # t, m
            path_sample = root_dir + f"sample_{idx}.png"
            plot_mesh(sample, mesh_pos, cells, n_t=5, path=path_sample)

            log['mesh_pos'] = mesh_pos

            if pl_module.use_embed:
                with open(root_dir + f"prompt_{idx}.txt", "w") as text_file:
                    text_file.write(prompt)
                log['prompt'] = prompt

            with open(root_dir + f"log_{idx}.pkl", "wb") as f:
                pickle.dump(log, f)

            if pl_module.use_embed:
                with open(root_dir + f"prompt_{idx}.txt", "w") as text_file:
                    text_file.write(prompt)
        

        loss = criterion_l2(log["samples"], log["inputs"])
        loss_l1 = criterion_l1(log["samples"], log["inputs"])
        all_losses.append(loss)
        all_losses_l1.append(loss_l1)
        all_times.append(end - start)

        if verbose:
            print("L2 Loss: ", loss)
            print("L1 Loss: ", loss_l1)
            print("Time: ", end - start)

    del all_times[0] # remove first time as it is usually an outlier

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    
    with open(root_dir + "losses_l1.pkl", "wb") as f:
        pickle.dump(all_losses_l1, f)

    with open(root_dir + "times.pkl", "wb") as f:
        pickle.dump(all_times, f)

    with open(root_dir + "mean_loss.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses))))
    
    with open(root_dir + "mean_loss_l1.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses_l1))))
    
    with open(root_dir + "mean_time.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_times))))

    print("Mean L2 Loss: ", torch.mean(torch.tensor(all_losses)))
    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses_l1)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))

def validate_cylinder(config, device):
    load_dir = config["load_dir"]
    dataconfig = config['data']
    verbose = config['verbose']
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval/"
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

    num_samples = len(valid_loader.dataset) # should be 100 

    criterion_l2 = ScaledLpLoss(p=2)
    criterion_l1 = torch.nn.L1Loss()

    plot_interval = 10
    all_losses = []
    all_losses_l1 = []
    all_times = []

    for idx in tqdm(range(0, num_samples)):
        if idx % plot_interval != 0:
            idx += 1
            continue
        batch = valid_loader.dataset.__getitem__(idx, eval=True)

        cells = batch.pop("cells")

        if pl_module.use_embed:
            prompt = batch.pop("prompt")

        batch = {k: v.unsqueeze(0).to(pl_module.device) for k, v in batch.items()}
        if pl_module.use_embed:
            batch["prompt"] = [prompt]

        start = time.time()
        with torch.no_grad():
            log = pl_module.log_images(batch, plot_diffusion_rows=False, plot_denoise_rows=False)
        end = time.time()

        pos = batch["pos"] # b, t, m, 3

        if "pad_mask" in batch.keys(): # need to trim padded data
            length = torch.sum(batch["pad_mask"][0], dtype=torch.long)
            log['inputs'] = log['inputs'][:, :, :length] # 1 t m c
            log['samples'] = log['samples'][:, :, :length] # 1 t m c
            pos = pos[:, :, :length] # b t m 3
        
        
        if idx % plot_interval == 0:

            mesh_pos = pos[0, 0, :, :2].detach().cpu() # m, 2

            inputs = log["inputs"][0, :, :, 0].detach().cpu() # t, m
            path_inputs = root_dir + f"inputs_{idx}.png" 
            plot_mesh(inputs, mesh_pos, cells, n_t=5, path=path_inputs)

            sample = log["samples"][0, :, :, 0].detach().cpu() # t, m
            path_sample = root_dir + f"sample_{idx}.png"
            plot_mesh(sample, mesh_pos, cells, n_t=5, path=path_sample)

            log['mesh_pos'] = mesh_pos

            if pl_module.use_embed:
                with open(root_dir + f"prompt_{idx}.txt", "w") as text_file:
                    text_file.write(prompt)
                log['prompt'] = prompt

            with open(root_dir + f"log_{idx}.pkl", "wb") as f:
                # save on memory
                del log["reconstruction"]
                pickle.dump(log, f)

            if pl_module.use_embed:
                with open(root_dir + f"prompt_{idx}.txt", "w") as text_file:
                    text_file.write(prompt)
        

        loss = criterion_l2(log["samples"], log["inputs"])
        loss_l1 = criterion_l1(log["samples"], log["inputs"])
        all_losses.append(loss)
        all_losses_l1.append(loss_l1)
        all_times.append(end - start)

        if verbose:
            print("L2 Loss: ", loss)
            print("L1 Loss: ", loss_l1)
            print("Time: ", end - start)

    del all_times[0] # remove first time as it is usually an outlier

    with open(root_dir + "losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    
    with open(root_dir + "losses_l1.pkl", "wb") as f:
        pickle.dump(all_losses_l1, f)

    with open(root_dir + "times.pkl", "wb") as f:
        pickle.dump(all_times, f)

    with open(root_dir + "mean_loss.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses))))

    print("Mean L2 Loss: ", torch.mean(torch.tensor(all_losses)))
    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses_l1)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))

def validate_ns2D(config, device):
    load_dir = config["load_dir"]
    dataconfig = config['data']
    verbose = config['verbose']
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval/"
    os.makedirs(root_dir, exist_ok=True)

    datamodule = FluidsDataModule(dataconfig)

    if "model_name" in config.keys() and config["model_name"] == "acdm":
        from modules.models.baselines.acdm import ACDM
        pl_module = ACDM(**config["model"],
                         normalizer=datamodule.normalizer,)
    else:
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

    start_idx = 0 # inclusive
    end_idx = 10 # exclusive

    plot_interval = 1
    all_losses = []
    all_times = []  

    print("starting at: ", start_idx)
    print("ending at: ", end_idx)
    print("Plot interval: ", plot_interval)

    idx = 0 
    prompt = None 
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            if idx < start_idx:
                idx += 1
                continue
                
            if idx >= end_idx:
                break

            if os.path.isfile(root_dir + f"losses_{idx+1}.pkl"):
                print("Loaded stats, Batch: ", idx)
                rec_loss = pickle.load(open(root_dir + f"losses_{idx+1}.pkl", "rb"))
                all_losses.append(rec_loss)
                idx += 1
                continue

            if "prompt" in batch.keys():
                prompt = batch.pop("prompt")

            batch = {k: v.to(pl_module.device) for k, v in batch.items()}
            
            if prompt is not None:
                batch["prompt"] = prompt

            start = time.time()
            log = pl_module.log_images(batch, N=batch_size, plot_diffusion_rows=False, plot_denoise_rows=False)
            end = time.time()

            if idx % plot_interval == 0:
                with open(root_dir + f"/log_{idx}.pkl", "wb") as f:
                    pickle.dump(log, f)

                inputs = log["inputs"][0, :, :, :, -1].detach().cpu() # t x y
                samples = log["samples"][0, :, :, :, -1].detach().cpu() # t, x y
                path_samples = root_dir + f"/samples_{idx}.png"
                plot_grid(inputs, samples, n_t=4, path=path_samples)

                if prompt is not None:
                    with open(root_dir + f"prompt_{idx}.txt", "w") as text_file:
                        text_file.write(prompt[0])

            rec_loss = F.l1_loss(log["inputs"], log["samples"])
            all_losses.append(rec_loss)
            all_times.append(end - start)
            idx += 1

            with open(root_dir + f"losses_{idx}.pkl", "wb") as f:
                pickle.dump(rec_loss, f)
            
            if verbose:
                print("Loss: ", rec_loss)
                print("Time: ", end - start)
    
    del all_times[0] # remove first time as it is usually an outlier

    with open(root_dir + "all_losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    
    with open(root_dir + "all_times.pkl", "wb") as f:
        pickle.dump(all_times, f)
    
    with open(root_dir + "mean_loss.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses))))

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))

def validate_ns2D_phiflow(config, device):
    ABLATE_PHIFLOW = True

    load_dir = config["load_dir"]
    dataconfig = config['data']
    batch_size = 1
    dataconfig['batch_size'] = batch_size
    root_dir = load_dir + "eval_phiflow"
    os.makedirs(root_dir, exist_ok=True)

    print("Running on device ", device)

    datamodule = FluidsDataModule(dataconfig)

    pl_module = LatentDiffusion(**config["model"],
                                normalizer=datamodule.normalizer,
                                use_embed=dataconfig["dataset"]["use_embed"])

    path = config["model_path"]

    checkpoint = torch.load(path, map_location=device)
    if "state_dict" in checkpoint.keys(): # sometimes the checkpoint is nested
        checkpoint = checkpoint["state_dict"]

    pl_module.load_state_dict(checkpoint)
    pl_module.eval()
    pl_module = pl_module.to(device)

    print("LDM Model loaded from: ", path)

    valid_loader = datamodule.val_dataloader()

    num_samples = 32*19

    plot_interval = 1 
    all_losses = []
    resolved_losses = []
    resolved_losses_halfres = []

    all_times = [] 
    all_times_sim = []
    all_times_sim_halfres = []

    print("Number of samples: ", num_samples)
    print("Plot interval: ", plot_interval)

    idx = 0 
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            checkpoint_file = root_dir + f"/losses_{idx+1}.pkl" # were saved after incrementing idx

            if os.path.isfile(checkpoint_file):
                rec_loss = pickle.load(open(checkpoint_file, "rb"))
                all_losses.append(rec_loss)

                resolved_loss = pickle.load(open(root_dir + f"/resolved_losses_{idx+1}.pkl", "rb"))
                resolved_losses.append(resolved_loss)

                idx += 1

                print(f"Loaded stats, Batch: {idx}, Rec Loss: {rec_loss}, Resolved Loss: {resolved_loss}")
                continue

            if pl_module.use_embed: 
                prompt = batch.pop("prompt")

            batch = {k: v.to(pl_module.device) for k, v in batch.items()}
            
            if pl_module.use_embed:
                batch["prompt"] = prompt

            start_time = time.time()
            log = pl_module.log_images(batch, N=batch_size, plot_diffusion_rows=False, plot_denoise_rows=False)
            end_time = time.time()

            log['inputs'] = log['inputs'].detach().cpu()
            log['samples'] = log['samples'].detach().cpu()
            buoyancy_y = batch['cond'].detach().cpu().squeeze().item()

            del log["reconstruction"]

            start_time_sim = time.time()
            solution_resolved = simulate_fluid(log['samples'], buoyancy_y) # assumes batch size of 1
            end_time_sim = time.time()
            log["solution_resolved"] = solution_resolved    

            if ABLATE_PHIFLOW:
                start_time_sim_halfres = time.time()
                solution_lowres = simulate_fluid_lowres(log['samples'], buoyancy_y, k=4)
                end_time_sim_halfres = time.time()
                log["solution_resolved_halfres"] = solution_lowres

            if idx % plot_interval == 0:

                with open(root_dir + f"/log_{idx}.pkl", "wb") as f:
                    pickle.dump(log, f)

                inputs = log["inputs"][0, :, :, :, -1].detach().cpu() # t x y
                samples = log["samples"][0, :, :, :, -1].detach().cpu() # t, x y
                path_samples = root_dir + f"/samples_{idx}.png"
                plot_grid(inputs, samples, n_t=4, path=path_samples)
                plot_grid(solution_resolved[0, :, :, :, -1], samples, n_t=4, path=root_dir + f"/resolved_{idx}.png")

                if pl_module.use_embed:
                    with open(root_dir + f"/prompt_{idx}.txt", "w") as text_file:
                        text_file.write(prompt[0])

            rec_loss = F.l1_loss(log["inputs"], log["samples"]).item()
            all_losses.append(rec_loss)

            resolved_loss = F.l1_loss(log["solution_resolved"], log["samples"]).item()
            resolved_losses.append(resolved_loss)

            if ABLATE_PHIFLOW:
                resolved_loss_halfres = F.l1_loss(log["solution_resolved_halfres"], log["solution_resolved"]).item()
                resolved_losses_halfres.append(resolved_loss_halfres)

            idx += 1

            with open(root_dir + f"/losses_{idx}.pkl", "wb") as f:
                pickle.dump(rec_loss, f)

            with open(root_dir + f"/resolved_losses_{idx}.pkl", "wb") as f:
                pickle.dump(resolved_loss, f)

            elapsed_time = round(end_time - start_time, 3)
            all_times.append(elapsed_time)
            sim_time = round(end_time_sim - start_time_sim, 3)
            all_times_sim.append(sim_time)

            sim_time_halfres = 0
            if ABLATE_PHIFLOW:
                sim_time_halfres = round(end_time_sim_halfres - start_time_sim_halfres, 3)
                all_times_sim_halfres.append(sim_time_halfres)
            print(f"Sim Time: {sim_time} seconds, Half Sim Time: {sim_time_halfres}, Sample Time: {elapsed_time} seconds, Batch: {idx}, Rec Loss: {rec_loss}, Resolved Loss: {resolved_loss}, HalfRes Loss: {resolved_loss_halfres} Buoyancy: {buoyancy_y}")

    with open(root_dir + "/all_losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    with open(root_dir + "/resolved_losses.pkl", "wb") as f:
        pickle.dump(resolved_losses, f)

    with open(root_dir + "/resolved_losses_halfres.pkl", "wb") as f:
        pickle.dump(resolved_losses_halfres, f)
    
    with open(root_dir + "/mean_loss.txt", "w") as text_file:
        text_file.write(str(torch.mean(torch.tensor(all_losses))))

    with open(root_dir + "/mean_resolved_loss.txt", "w") as text_file: 
        text_file.write(str(torch.mean(torch.tensor(resolved_losses))))

    print("Mean L1 Loss: ", torch.mean(torch.tensor(all_losses)))
    print("Mean Resolved L1 Loss: ", torch.mean(torch.tensor(resolved_losses)))

    with open(root_dir + "/all_times.pkl", "wb") as f:
        pickle.dump(all_times, f)
    
    with open(root_dir + "/all_times_sim.pkl", "wb") as f:
        pickle.dump(all_times_sim, f)

    with open(root_dir + "/all_times_sim_halfres.pkl", "wb") as f:
        pickle.dump(all_times_sim_halfres, f)
    
    print("Mean Time: ", torch.mean(torch.tensor(all_times)))
    print("Mean Sim Time: ", torch.mean(torch.tensor(all_times_sim)))

def main(args):
    config=get_yaml(args.config)
    config['verbose'] = False
    mode = config['data']['mode'] # get mode
    device = args.device
    ddim_steps = args.ddim_steps

    if mode == "cylinder":
        if ddim_steps > 0:
            validate_cylinder_ddim(config, device, ddim_steps)
        else:
            validate_cylinder(config, device)
    elif mode == "ns2D":
        if "phiflow" in config.keys() and config["phiflow"]:
            validate_ns2D_phiflow(config, device)
        else:
            validate_ns2D(config, device)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate an LDM')
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", default=False) 
    parser.add_argument("--ddim_steps", default=0, type=int)
    args = parser.parse_args()

    main(args)