# Training and Inference
For more information about the relevant training parameters, see the [configs](configs) directory.

## Autoencoder
To train an autoencoder with only KL regularization (No GAN or LPIPS):
```
python ae/train_AE_KL.py --config=path/to/config
```

To train an autoencoder with all possible methods defined by a config:
```
python ae/train_AE.py --config=path/to/config
```

## Latent Diffusion Model
To train a latent diffusion model:
```
python ldm/train_ldm.py --config=path/to/config
```

## Baselines
To train a baseline model for the cylinder flow problem:
```
python baselines/cylinder/train_{model}.py --config=path/to/config
```
To train a baseline model for the smoke buoyancy (ns2D) problem:
```
python baselines/ns2D/train_{model}.py --config=path/to/config
```
Note that the FNO, Unet, and Resnet models all use the same script.

## Validation/Inference
To generate reconstructed samples on the validation set and evaluate a mean reconstruction loss:
```
python validation/validate_AE.py --config=path/to/config
```

For baselines (not including ACDM), to generate predicted samples on the validation set and evaluate a mean prediction loss:
```
python validation/validate_{cylinder/ns2D}.py --config=path/to/config
```

For LDM and ACDM models, to conditionally sample from the validation set and evaluate and mean prediction loss:
```
python validation/validate_ldm.py --config=path/to/config
```

To count the # of FLOPs and generate a profile:
```
python validation/profile_flops.py --config=path/to/config
```