#!/bin/bash
#SBATCH --job-name=seg-guided-diff
#SBATCH --output=seg-guided-diff.out
#SBATCH --error=seg-guided-diff.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
# gpu 
#SBATCH --gres=gpu:2

ml python/anaconda3

source deactivate
source activate py312

python3 main.py --mode train \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset amos_mri_labeled \
    --img_dir /vol/aimspace/projects/practical_WS2425/diffusion/data/amos_slices/images \
    --seg_dir /vol/aimspace/projects/practical_WS2425/diffusion/data/amos_slices/labels \
    --img_name_filter /vol/aimspace/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/utils/mask_MRI_min2cls.csv \
    --model_type DDIM \
    --segmentation_guided \
    --segmentation_channel_mode single \
    --num_segmentation_classes 16 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 100 \
    # --resume_epoc 5 \
    # --use_ablated_segmentations \
    # --eval_noshuffle_dataloader \
    # --eval_mask_removal \
    # --eval_blank_mask \
    # --eval_sample_size x \
