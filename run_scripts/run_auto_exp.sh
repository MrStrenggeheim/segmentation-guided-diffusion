#!/bin/bash
#SBATCH --job-name=test-seg-guided-diff-exp
#SBATCH --mail-user=florian.hunecke@tum.de
#SBATCH --mail-type=ALL
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# make sure to not get interrupted
#SBATCH --qos=master-queuesave
##SBATCH --partition=universe,asteroids

ml python/anaconda3

source deactivate
source activate py312

# python -m debugpy --listen 5678 --wait-for-client main.py \
python main.py \
    --mode train \
    --img_size 64 \
    --num_img_channels 1 \
    --dataset amos_ct_axial \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels \
    --model_type DDIM \
    --index_range 0 200 \
    --segmentation_guided \
    --segmentation_ingestion_mode mul \
    --segmentation_channel_mode single \
    --num_segmentation_classes 73 \
    --train_batch_size 32 \
    --eval_batch_size 4 \
    --num_epochs 10 \
    --resume \
    # --img_type CT \
    # --load_images_as_np_arrays \

    # --resume_epoc 2 \
    # --img_name_filter /vol/aimspace/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/utils/mask_CT.csv \
    # --use_ablated_segmentations \
    # --eval_noshuffle_dataloader \
    # --eval_mask_removal \
    # --eval_blank_mask \
    # --eval_sample_size x \
