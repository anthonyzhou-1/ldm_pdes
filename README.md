# Latent Diffusion for Fluids

## Requirements

To install requirements:

```setup
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 torchdata=0.6.0 -c pytorch -c nvidia
conda install pytorch-scatter -c pyg
conda install lightning wandb h5py tensorboard transformers -c conda-forge
pip install einops open3d sparse-dot-mkl timm scikit-image
```

If you cannot install pytorch<=2.0.1, please refer to the [Compatibility](#compatibility) section, as some libraries require this version. 

## Datasets
- Cylinder_Flow
    - 1000/100 train/valid samples
    - Incompressible NS in water, Re ~100-1000, dt = 0.01
    - (n_samples, time, n_nodes, quantity)
    - contains information about pressure, velocities, position of mesh points, connectivity of mesh

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