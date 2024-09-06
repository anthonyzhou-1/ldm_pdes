# Configs
The config files are organized by the problem of interest {cylinder/ns2D} and the training task {ae/baselines/ldm}
All config files will expect:
```
- data_dir: A path to the dataset.h5
- stat_path: A path to the normalized statistics if using the normalizer 
    - This does not need to be generated beforehand, if the file does not exist it will be generated and stored there
- default_root_dir: A default log directory (logs/)
- wandb: A wandb project and name
```

Additionally, when loading model for evaluation/inference (validation_{}.py), the scripts will expect:
```
load_dir: directory to save evaluation outputs (images, plots, log files, etc.)
model_path: path to pretrained model to evaluate (/path/to/model.ckpt)
model_name: name of model (gino/acdm/unet/ldm/etc.)
```

## Autoencoder Configs
Autoencoder configs can vary based on if the mesh encoder/decoder is used or not. Furthermore, there are additional options for enabling GAN/LPIPS training. If LPIPS is enabled, the config will expect a file path to a pretrained DPOT [model](https://huggingface.co/hzk17/DPOT). Some important parameters:
```
- padding: set to true if using batch size larger than 1 in mesh problems. This is because each data sample is a different shape and needs to be padded to max_len to stack samples together. 
- hidden_channels: main way to scale the model
- gno_radius: defines radius of GNO kernel. Larger values will quickly take more memory + compute.
- z_channels: channel dimension of the latent space
- ch_mult: defines number of down/upsampling layers (len(ch_mult)). Also expands the channel dimension by ch_mult[i] * hidden_channels at each down/upsampling layer.
- kl_weight: defines strength of KL regularization.
- disc_weight: defines strength of discriminator losses. If disc_weight=0, discriminator is turned off.
- perceptual_weight: defines strength of perceptual loss. If perceptual_weight=0, LPIPS is turned off.
```

## LDM Configs
LDM configs expect a path to a pretrained autoencoder, and has additional options for conditioning and backbone parameters. Some important parameters:
```
- use_embed: set to true if using text-conditioning
- dist: set to true if using DDP to sync logging statistics across devices.
- hidden_size: hidden dimension of DiT backbone. Main way to scale the model.
- pretrained_path: path to pretrained autoencoder.
- conditional: set to False if doing unconditional diffusion
- clip_denoised, parameterization, scale_factor, beta_schedule: controls different modifications to the denoising process
```

## Baseline Configs
Baseline configs outline different benchmark models and parameters. Each model has its own hyperparameters, but in general the size is controlled by scaling a hidden dimension.

