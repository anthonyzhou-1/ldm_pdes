import argparse
from datetime import datetime
import torch
import os 

from modules.utils import get_yaml, save_yaml 
from modules.modules.callbacks import GridPlottingCallback, MeshPlottingCallback
from dataset.datamodule import FluidsDataModule

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    config=get_yaml(args.config)
    aeconfig = config['model']['aeconfig']
    lossconfig = config['model']['lossconfig']
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
        monitor="val/total_loss", # track total loss (reconstruction, KL, discriminator, LPIPS, etc.)
        filename= "model_{epoch:02d}-{val/total_loss:.2f}",
        dirpath=path,
        save_top_k=1,
        save_last=True,
    )
    
    datamodule = FluidsDataModule(dataconfig)

    if dataconfig["mode"] == "cylinder":
        from modules.models.ae.ae_mesh import Autoencoder
        eval_callback = MeshPlottingCallback()
    else:
        from modules.models.ae.ae_grid import Autoencoder
        eval_callback = GridPlottingCallback()

    model = Autoencoder(aeconfig, 
                        lossconfig, 
                        trainconfig,
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
                        strategy=trainconfig["strategy"],
                        limit_train_batches=int(trainconfig["limit_train_batches"]) if "limit_train_batches" in trainconfig else 1.0,
                        limit_val_batches=int(trainconfig["limit_val_batches"]) if "limit_val_batches" in trainconfig else 1.0,)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                    datamodule=datamodule,
                    ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a AE')
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    main(args)