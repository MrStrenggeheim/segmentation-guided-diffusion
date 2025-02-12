import sys
from dataclasses import dataclass

import diffusers
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

# Add custom imports from the parent directory (if needed)
sys.path.append("..")
from eval import (
    SegGuidedDDIMPipeline,
    SegGuidedDDPMPipeline,
    add_segmentations_to_noise,
    evaluate,
)


@dataclass
class SegGuidedDiffConfig:
    model_type: str = "DDPM"  # "DDPM" or "DDIM"
    image_size: int = 256
    num_img_channels: int = 1
    batch_size: int = 16
    num_epochs: int = 200
    learning_rate: float = 1e-5
    lr_warmup_steps: int = 500
    segmentation_guided: bool = False
    segmentation_ingestion_mode: str = "concat"  # "concat", "add", "mul", "replace"
    num_segmentation_classes: int = None  # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume: bool = False
    load_images_as_np_arrays: bool = False
    output_dir: str = None

    @classmethod
    def from_args(cls, args):
        # Compute output directory similarly to your previous implementation.
        output_dir = "output/{}-{}-{}-{}".format(
            args.model_type.lower(),
            args.img_size,
            args.num_img_channels,
            args.segmentation_ingestion_mode,
        )
        if args.segmentation_guided:
            output_dir += "-segguided"
        if args.use_ablated_segmentations:
            output_dir += "-ablated"
        return cls(
            model_type=args.model_type,
            image_size=args.img_size,
            num_img_channels=args.num_img_channels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=1e-5,  # or override with an additional arg if desired
            lr_warmup_steps=500,
            segmentation_guided=args.segmentation_guided,
            num_segmentation_classes=args.num_segmentation_classes,
            use_ablated_segmentations=args.use_ablated_segmentations,
            resume=args.resume,
            load_images_as_np_arrays=args.load_images_as_np_arrays,
            output_dir=output_dir,
        )


class SegGuidedDiff(pl.LightningModule):
    def __init__(self, config: SegGuidedDiffConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Determine input channels (if segmentation-guided, add extra channel if needed)
        in_channels = config.num_img_channels
        if config.segmentation_guided:
            if config.segmentation_ingestion_mode == "concat":
                in_channels += 1

        # Define the UNet model from diffusers.
        self.model = diffusers.UNet2DModel(
            sample_size=config.image_size,
            in_channels=in_channels,
            out_channels=config.num_img_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        # Set up the noise scheduler based on the chosen model type.
        if config.model_type == "DDPM":
            self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
        elif config.model_type == "DDIM":
            self.noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
        else:
            raise ValueError("Unsupported model_type. Choose either 'DDPM' or 'DDIM'.")

        self.learning_rate = config.learning_rate

    def forward(self, x, timesteps):
        # Forward pass through the UNet to predict noise residual.
        return self.model(x, timesteps, return_dict=False)[0]

    def training_step(self, batch, batch_idx):
        device = self.device
        clean_images = batch["images"].to(device)
        noise = torch.randn_like(clean_images)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        if self.config.segmentation_guided:
            noisy_images = add_segmentations_to_noise(
                noisy_images, batch, self.config, device
            )

        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # only called once
        device = self.device
        clean_images = batch["images"].to(device)
        noise = torch.randn_like(clean_images)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        if self.config.segmentation_guided:
            noisy_images = add_segmentations_to_noise(
                noisy_images, batch, self.config, device
            )

        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            self.eval_and_save(
                self.config,
                self.model,
                self.noise_scheduler,
                batch,
                epoch=self.current_epoch,
                step=self.global_step,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def eval_and_save(self, config, model, noise_scheduler, batch, epoch, step):
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(
                    unet=model,
                    scheduler=noise_scheduler,
                    eval_dataloader=None,
                    external_config=config,
                )
            else:
                pipeline = diffusers.DDPMPipeline(unet=model, scheduler=noise_scheduler)
        elif config.model_type == "DDIM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model,
                    scheduler=noise_scheduler,
                    eval_dataloader=None,
                    external_config=config,
                )
            else:
                pipeline = diffusers.DDIMPipeline(unet=model, scheduler=noise_scheduler)

        model.eval()

        if config.segmentation_guided:
            evaluate(config, epoch, pipeline, batch, step=step)
        else:
            evaluate(config, epoch, pipeline, step=step)

        model.train()
