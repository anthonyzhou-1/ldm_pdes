# Latent Diffusion for Fluids

## Requirements

To install requirements:

```setup
conda create -n "my_env" 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 torchdata=0.6.0 -c pytorch -c nvidia
conda install pytorch-scatter -c pyg
conda install lightning wandb h5py tensorboard transformers -c conda-forge
pip install einops open3d sparse-dot-mkl timm scikit-image
```

If you cannot install pytorch<=2.0.1, please refer to the [Compatibility](#compatibility) section, as some libraries require this version. 

## Datasets
Full datasets are available [here.]()

- Cylinder_Flow
    - 1000/100 train/valid samples
    - Incompressible NS in water, Re ~100-1000, dt = 0.01
    - Around 2000 mesh points, downsampled to 25 timesteps
    - Each data sample has a different shape, so they cannot be stacked. Therefore each data sample is in its own numbered dictionary ('0' has sample 0, '1' has sample 1, etc.). 
        - dataset.h5 (keys: '0', '1', ... etc.)
            - '0' (keys: 'cells', 'mesh_pos', 'metadata', 'node_type', 'pressure', 'u', 'v')
                - 'cells': shape (num_edges, 3). Defines connectivity in triangular mesh. Only used for plotting
                - 'mesh_pos': shape (num_nodes, 2). Defines the position of each node in the mesh. 
                - 'node_type': shape (num_nodes, 1). Defines type of each node (0=fluid, 4=inlet, 5=outlet, 6=boundaries/walls)
                - 'pressure': shape (num_timesteps, num_nodes, 1). Defines pressure at each timestep for all mesh points.
                - 'u': shape (num_timesteps, num_nodes). Defines x-component of velocity at each timestep for all mesh points.
                - 'v': shape (num_timesteps, num_nodes). Defines y-component of velocity at each timestep for all mesh points.
                - 'metadata': (keys: 'center', 'domain_x', 'domain_y', 'prompt', 'radius', 'reynolds_number', 't_end', 'u_inlet', 'v_inlet')
                    - 'center': shape (2,). Extracted center of cylinder, in meters.
                    - 'domain_x': shape (2,). Bounds of x in the domain, in meters.
                    - 'domain_y': shape (2,). Bounds of y in the domain, in meters.
                    - 'prompt': shape(). Procedurally generated prompt using template in paper. Read with ['prompt'].asstr()[()].
                    - 'radius': shape (). Extracted radius if cylinder, in meters. 
                    - 'reynolds_number': shape (). Extracted Reynolds number of simulation.
                    - 't_end': shape (). Final time of simulation.
                    - 'u_inlet': shape(). x-component of velocity at the inlet.
                    - 'v_inlet': shape(). y-component of velocity at the inlet.
            - '1', '2', ... etc.
- Smoke Buoyancy (NS2D)
    - 2496/608 train/valid samples.
        - Datasets are divided into separates files with 32 samples each. This results in 78 training files (78x32=2496) and 19 valid files (19x32=608)
    - Smoke driven by a buoyant force, dt=1.5
    - 128x128 spatial resolution, with 56 timesteps.
    - Each file contains 32 samples for a given seed, with uniform shape. The text captions are not uniform, so they are stored in a numbered dictionary as well.
        - dataset.h5 (keys: 'train' or 'valid')
            - 'train' (keys: 'buo_y', 'dt', 'dx', 'dy', 't', 'text_labels', 'u', 'vx', 'vy', 'x', 'y')
                - 'buo_y': shape (32,). Contains a scalar buoyancy factor for each sample.
                - 'dt', 'dx', 'dy': shape (32,). Contains a scalar dt, dx, or dy for each sample.
                - 't': shape (32, num_timesteps). Contains the time at each timestep for each sample.
                - 'vx': shape (32, num_timesteps, resolution_x, resolution_y). Contains the x-component of velocity at each nodal position, timestep, and sample. 
                - 'y': shape (32, num_timesteps, resolution_x, resolution_y). Contains the y-component of velocity at each nodal position, timestep, and sample. 
                - 'u': shape (32, num_timesteps, resolution_x, resolution_y). Contains the smoke density at each nodal position, timestep, and sample. 
                - 'x': shape (32, resolution_x). Contains the x-position for each position along the x-axis for each sample.
                - 'y': shape (32, resolution_y). Contains the y-position for each position along the y-axis for each sample.
                - 'text_labels' (keys: '0', '1', ..., '31')
                    - '0': shape (). Contains the text caption for the 0-th sample. Read with ['text_labels']['0'].asstr()[()]
                    - '1', '2', ... '31': Contains text cpation for the n-th sample.

## Training an Autoencoder
The config file will expect:
- A data directory with the formatted data
- A log directory to save model checkpoints and evaluation images/videos
- If using LPIPS (only implemented for grid data), a model checkpoint to load
- If using wandb for logging, a working wandb instance

### Grid Data (Not tested yet)

```
python train_AE.py --config=configs/structured/ae_grid.yaml
```

### Mesh Data

Depending on the config, this could require between 10-40 GB of GPU memory.
```
python train_ae_gino.py --config=configs/structured/ae_gino_32x8x8x8.yaml
```

## Training a Latent Diffusion model
The config file will expect:
- A data directory with the formatted data
- A log directory to save model checkpoints and evaluation images/videos
- A model checkpoint for the pretrained autoencoder
- If using wandb for logging, a working wandb instance

### Unconditional Diffusion
```
python train_ldm.py --config=configs/ldm/ldm_32x8x8x8.yaml
```

### Conditional Diffusion (Doesn't work yet)
```
python train_ldm.py --config=configs/ldm/ldm_32x8x8x8_cond.yaml
```

## Compatibility
Some parts of the code relies on [Open3D](https://www.open3d.org/). Specifically, Open3D requires a version of torch <=2.0.1; this option can be disabled in the config files if the installation is not compatible, and the codebase can fall back to a native PyTorch implementation. This is slower and requires more memory, but can be set with the flag use_open3d=False in all configs.

Additionally, there are certain reports of FFT failing for pytorch-cuda <=11.7 ([issue](https://github.com/pytorch/pytorch/issues/88038)). Only the FNO and GINO baselines make use of FFT.

Lastly, the smoke buoyancy problem relies on the torchdata and datapipes [package](https://github.com/pytorch/data), which will be deprecated in the future. This may also cause compatibility issues with newer versions of torch (>=2.0.1), specifically:

```
File "/home/anaconda3/envs/env-name/lib/python3.11/site-packages/torchdata/datapipes/iter/util/cacheholder.py", line 24, in <module>
    from torch.utils._import_utils import dill_available
ModuleNotFoundError: No module named 'torch.utils._import_utils'
```

A workaround is to define a function to always return false:

```
cd /path-to-conda-env/lib/python3.11/site-packages/torch/utils
echo "def dill_available(): return False" > _import_utils.py
```

## SLURM Users
For those leveraging multiprocessing on a SLURM cluster, there are some additional considerations:
- If you plan on training the model with text capabilities, it is recommended to manually download the pretrained LLM weights (RoBERTa) and load them locally, as downloading weights on the fly may cause the script to hang. 

```
# On the fly. Might cause the script to hang.
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base") 
model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

 # Loading weights locally. Safer option.
tokenizer = AutoTokenizer.from_pretrained(cache_path, local_files_only=True)
model = RobertaModel.from_pretrained(cache_path, local_files_only=True)
```

- You may need to limit the number of train/val batches per epoch if using datapipes. In some DDP cases, having incomplete batches can cause GPUs to hang. [Issue](https://github.com/Lightning-AI/pytorch-lightning/issues/11910)