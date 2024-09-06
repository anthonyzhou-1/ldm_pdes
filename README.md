# Latent Diffusion for Fluids

## Requirements

To install requirements:

```setup
conda create -n "my_env" 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 torchdata=0.6.0 -c pytorch -c nvidia
conda install pytorch-scatter -c pyg
conda install lightning wandb h5py tensorboard transformers -c conda-forge
pip install einops open3d sparse-dot-mkl timm 
```

Optional installs for image captioning and FLOPs profiling:
```
pip install scikit-image deepspeed
```

If you cannot install pytorch<=2.0.1, please refer to the [Compatibility](#compatibility) section, as some libraries require this version. 

## Datasets
Full datasets are available [here.]()
Please refer to the [dataset](dataset) directory for a description of the raw data and dataloading. 

## Training and Inference
Please refer to the [scripts](scripts) directory. Also, the [configs](configs) directory has details on different training/inference settings. 

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