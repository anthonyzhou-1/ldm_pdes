import argparse
from datetime import datetime
import torch
import os 

from modules.utils import get_yaml, save_yaml 
from modules.modules.callbacks import NS2DCallback
from dataset.datamodule import FluidsDataModule
from modules.models.baselines.ns2D import NS2DModule

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('high')

def main(args):
    config=get_yaml(args.config)
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)

    name = config["wandb"]["name"] + now
    wandb_logger = WandbLogger(project=config["wandb"]["project"],
                               name=name,)
    path = trainconfig["default_root_dir"] + name + "/"

    if torch.cuda.current_device() == 0:
        os.makedirs(path, exist_ok=True) 
        save_yaml(config, path + "config.yml")
        print("Making folder on rank 0")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/rollout_error",
        filename= "model_{epoch:02d}-{val/rollout_error:.3f}",
        dirpath=path,
        save_top_k=1,
    )
    
    datamodule = FluidsDataModule(dataconfig)
    eval_callback = NS2DCallback()

    model = NS2DModule(modelconfig=modelconfig,
                       trainconfig=trainconfig,
                       normalizer=datamodule.normalizer,
                       batch_size=dataconfig["batch_size"],
                       accumulation_steps=trainconfig["accumulate_grad_batches"],)

    trainer = L.Trainer(devices = trainconfig["devices"],
                        accelerator = trainconfig["accelerator"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps=trainconfig["log_every_n_steps"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        callbacks=[checkpoint_callback, eval_callback],
                        logger=wandb_logger,
                        strategy=trainconfig["strategy"],)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                    datamodule=datamodule,
                    ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a baseline')
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    main(args)