import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

# Add custom imports from the parent directory
sys.path.append("..")

from model import DiffusionLightningModule, DiffusionModelConfig

from utils.dataset.amos import AMOSDataModule
from utils.utils import get_last_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="diffusion_model")
    parser.add_argument("--log_dir", type=str, default="tb_logs")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--seg_dir", type=str, default=None)
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_img_channels", type=int, default=1)
    parser.add_argument("--img_type", type=str, default=None)
    parser.add_argument("--index_range", type=int, nargs=2, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["DDPM", "DDIM"], default="DDPM"
    )
    parser.add_argument("--segmentation_guided", action="store_true")
    parser.add_argument("--segmentation_ingestion_mode", type=str, default="concat")
    parser.add_argument("--num_segmentation_classes", type=int, default=None)
    parser.add_argument("--load_images_as_np_arrays", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_ablated_segmentations", action="store_true")
    args = parser.parse_args()

    # Set random seed and precision
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")

    # Set up the data module (reuse your existing AMOSDataModule)
    data_module = AMOSDataModule(
        img_dir=args.img_dir,
        seg_dir=args.seg_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        transforms=args.transforms,
        num_workers=args.num_workers,
    )

    # Set up TensorBoard logger and log hyperparameters
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)
    logger.log_hyperparams(vars(args))

    # Set up a ModelCheckpoint callback to monitor the validation loss
    checkpoint_callback = ModelCheckpoint(
        filename=f"{args.exp_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_last=True,
        every_n_train_steps=1000,
    )

    # Build the training configuration from the argparse arguments
    cfg = DiffusionModelConfig.from_args(args)
    # Instantiate your diffusion model LightningModule with the config
    model = DiffusionLightningModule(cfg)

    # Set up the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=16,
        val_check_interval=500,
        limit_val_batches=1,
        callbacks=[checkpoint_callback],
        fast_dev_run=False,
        gradient_clip_val=1.0,
    )

    # Print a model summary
    # summary(model, (args.num_img_channels, args.img_size, args.img_size), depth=10)

    # Optionally resume training if a checkpoint exists
    last_ckpt = None
    if args.resume:
        last_ckpt = get_last_checkpoint(os.path.join(args.log_dir, args.exp_name))
        if last_ckpt:
            print(f"Resuming from checkpoint: {last_ckpt}")
        else:
            print("No checkpoint found. Training from scratch.")

    trainer.fit(model, data_module, ckpt_path=last_ckpt)
