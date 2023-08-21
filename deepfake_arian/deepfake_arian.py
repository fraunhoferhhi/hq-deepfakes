import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from .losses import DeepfakeLoss
from .dataset import DeepfakeDataset
from .model import custom

# MODEL
class DeepfakeModel(pl.LightningModule):
    def __init__(self,
                 cfg: str = "EB4_RDA",
                 lr: float = 5e-5,
                 eps: float = 1e-7,
                 l2_weight: float = 1,
                 l1_weight: float = 0,
                 ffl_weight: float = 0,
                 stop_warping: int = 125000,
                 image_logging_interval: int = 1000):
        
        super().__init__()

        self.cfg = cfg
        self.lr = lr
        self.eps = eps
        self.stop_warping = stop_warping
        self.image_logging_interval = image_logging_interval

        self.ae = custom(cfg)

        self.loss_fn = DeepfakeLoss(3, 2, True, l2_weight=l2_weight, l1_weight=l1_weight, ffl_weight=ffl_weight)

    def forward(self, x):
        return self.ae(x)

    def training_step(self, batch, batch_idx):
        X_a, y_a, m_a = batch["A"]
        X_b, y_b, m_b = batch["B"]

        if self.stop_warping < self.global_step:
            X = torch.cat((X_a, X_b), dim = 1)
        else:
            X = torch.cat((y_a, y_b), dim = 1)
            
        pred_a, pred_b = self.forward(X)
        
        loss_a = self.loss_fn.compute(pred_a, y_a, m_a)
        loss_b = self.loss_fn.compute(pred_b, y_b, m_b)

        loss = 0.

        for key in loss_a.keys():
            loss += (loss_a[key] + loss_b[key])

            self.log("{}/A".format(key), loss_a[key])
            self.log("{}/B".format(key), loss_b[key])
            self.log("{}/Full".format(key), loss_a[key]+loss_b[key])

        self.log("Loss", loss)

        if self.global_step % self.image_logging_interval == 0:
            self.logger.log_image(key="Recon vs. Target", images=[torch.concat((pred_a[0, [2,1,0]], y_a[0, [2,1,0]], pred_b[0, [2,1,0]], y_b[0, [2,1,0]]), 2)])


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.lr, eps=self.eps)
        return optimizer



# DATAMOUDLE
class DeepfakeDatamodule(pl.LightningDataModule):
    '''
    Lightningdatamodule for Deepfake training. Loads one dataset per id and uses a combined loader.
    params:
        batch_size: batch size for loaders, is split in half and distributed over the 2 loaders
        num_workers: num workers, works analogue to batch size
        path_a: path to directory that holds faceset of person a
        path_b: see above
        input_size: size of images in path_a & path_b
        model_img_size: size of in and outputs of autoencoder
        coverage_ratio: center coverage of images to crop before augmenting etc.
        no_flip: set true to disable flipping augmentation
    '''
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 path_a: str,
                 path_b: str,
                 input_size: int = 512,
                 model_img_size: int = 256,
                 coverage_ratio: float = 0.8,
                 no_flip: bool = False):
        
        super().__init__()

        # save attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_a = path_a
        self.path_b = path_b
        self.input_size = input_size
        self.model_img_size = model_img_size
        self.coverage_ratio = coverage_ratio
        self.no_flip = no_flip

    def train_dataloader(self):
        ds_a = DeepfakeDataset(self.path_a, self.input_size, self.model_img_size, self.coverage_ratio, self.no_flip)
        ds_b = DeepfakeDataset(self.path_b, self.input_size, self.model_img_size, self.coverage_ratio, self.no_flip)
        
        dl_a = DataLoader(ds_a, batch_size=self.batch_size//2, num_workers=self.num_workers//2, drop_last=True, shuffle=True)
        dl_b = DataLoader(ds_b, batch_size=self.batch_size//2, num_workers=self.num_workers//2, drop_last=True, shuffle=True)

        iterable = {"A": dl_a,
                    "B": dl_b}
        
        loader = CombinedLoader(iterable, mode="max_size_cycle")

        return loader
