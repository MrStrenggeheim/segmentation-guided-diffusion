#!/bin/bash
#SBATCH --job-name=sgd-amos_both_base
#SBATCH --mail-user=florian.hunecke@tum.de
#SBATCH --mail-type=ALL
#SBATCH --output=logs/amos_both_base.out
#SBATCH --error=logs/amos_both_base.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# make sure to not get interrupted
SBATCH --qos=master-queuesave
##SBATCH --partition=universe,asteroids

ml python/anaconda3

source deactivate
source activate py312

python main.py \
    --mode train \
    --img_size 128 \
    --num_img_channels 1 \
    --dataset amos_both_base \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels \
    --model_type DDIM \
    --segmentation_guided \
    --segmentation_ingestion_mode concat \
    --segmentation_channel_mode single \
    --num_segmentation_classes 73 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 100 \
    --transforms "['ToTensor', 'Resize', 'CenterCrop', 'Normalize']" \
    --resume \
    # --img_type MRI \
    # --lr 0.0001 \
    # --index_range 0 200 \
    # --model_type DDIM \
    # --load_images_as_np_arrays \

    # --resume_epoc 2 \
    # --img_name_filter /vol/aimspace/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/utils/mask_CT.csv \
    # --use_ablated_segmentations \
    # --eval_noshuffle_dataloader \
    # --eval_mask_removal \
    # --eval_blank_mask \
    # --eval_sample_size x \
