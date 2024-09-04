import argparse
from datetime import datetime
import torch
import os 

from modules.utils import get_yaml, save_yaml
from dataset.datamodule import FluidsDataModule
from modules.models.ddpm import LatentDiffusion
from modules.modules.callbacks import MeshLDMCallback, EMA, GridLDMCallback

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    config=get_yaml(args.config)
    trainconfig = config['training']
    dataconfig = config['data']
    modelconfig = config['model']

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
    
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            filename= "model_{epoch:02d}-{val/loss:.2f}",
            dirpath=path,
            save_top_k=1,
            save_last=True
        )]
    
    if dataconfig["mode"] == "ns2D":
        callbacks.append(GridLDMCallback())
    else:
        callbacks.append(MeshLDMCallback())

    if trainconfig["ema_decay"] is not None:
        ema_callback = EMA(decay=trainconfig["ema_decay"],
                        every_n_steps=trainconfig["ema_every_n_steps"])
        print("Using EMA with decay: ", trainconfig["ema_decay"])
        callbacks.append(ema_callback)

    # setup scheduler config
    if "scheduler_config" in modelconfig.keys():
        modelconfig['scheduler_config']["batch_size"] = dataconfig["batch_size"]
        modelconfig['scheduler_config']["accumulate_grad_batches"] = trainconfig["accumulate_grad_batches"]
        modelconfig['scheduler_config']['dataset_size'] = trainconfig["dataset_size"]
        modelconfig['scheduler_config']['max_epochs'] = trainconfig["max_epochs"]

    datamodule = FluidsDataModule(dataconfig)
    
    model = LatentDiffusion(**modelconfig,
                            normalizer=datamodule.normalizer,
                            use_embed=dataconfig["dataset"]["use_embed"])

    trainer = L.Trainer(devices = trainconfig["devices"],
                        accelerator = trainconfig["accelerator"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps=trainconfig["log_every_n_steps"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        callbacks=callbacks,
                        logger=wandb_logger,
                        strategy=trainconfig["strategy"],
                        accumulate_grad_batches=trainconfig["accumulate_grad_batches"],
                        gradient_clip_val=trainconfig["gradient_clip_val"] if "gradient_clip_val" in trainconfig else 0,
                        limit_train_batches=trainconfig["limit_train_batches"] if "limit_train_batches" in trainconfig else 1.0,
                        limit_val_batches=trainconfig["limit_val_batches"] if "limit_val_batches" in trainconfig else 1.0,)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LDM')
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    main(args)