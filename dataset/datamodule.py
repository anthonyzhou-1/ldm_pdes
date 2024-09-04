import lightning as L
from torch.utils.data import DataLoader
from dataset.cylinder import CylinderMeshDataset
from dataset.smoke_data import train_datapipe_ns_cond, valid_datapipe_ns_cond
from modules.modules.normalizer import Normalizer
from modules.utils import Struct

class FluidsDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        dataset_config = dataconfig["dataset"]
        normalizer_config = dataconfig["normalizer"]
        
        self.data_dir = dataconfig["data_dir"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.mode = dataconfig["mode"]
        self.normalizer = None

        if "drop_last" in dataconfig.keys():
            self.drop_last = dataconfig["drop_last"]
        else:
            self.drop_last = False

        if self.mode == "ns2D":
            self.train_dataset = train_datapipe_ns_cond(Struct(**dataset_config)) # change dict to object to support dot notation
            self.val_dataset = valid_datapipe_ns_cond(Struct(**dataset_config))
            
        elif self.mode == "cylinder":
            self.train_dataset = CylinderMeshDataset(data_dir = self.data_dir + "train_downsampled_labeled.h5",
                                                    **dataset_config)
            self.val_dataset = CylinderMeshDataset(data_dir = self.data_dir + "valid_downsampled_labeled.h5",
                                                    **dataset_config)
            
        self.normalizer = Normalizer(dataset=self.train_dataset,
                                     **normalizer_config)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          drop_last=self.drop_last)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None