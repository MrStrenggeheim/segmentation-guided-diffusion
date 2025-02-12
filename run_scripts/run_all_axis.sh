#!/bin/bash
#SBATCH --job-name=seg-guided-diff-all
#SBATCH --mail-user=florian.hunecke@tum.de
#SBATCH --mail-type=ALL
#SBATCH --output=seg-guided-diff-all.out
#SBATCH --error=seg-guided-diff-all.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --qos=master-queuesave
##SBATCH --partition=universe,asteroids

ml python/anaconda3

source deactivate
source activate py312

python3 main.py \
    --mode eval \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset amos_ct_all_axis \
    --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/test_data/images \
    --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/test_data/labels \
    --model_type DDIM \
    --segmentation_guided \
    --segmentation_ingestion_mode concat \
    --segmentation_channel_mode single \
    --num_segmentation_classes 73 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_epochs 100 \
    --transforms "['ToTensor', 'Resize', 'CenterCrop', 'Normalize']" \
    # --img_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/images_all_axis \
    # --seg_dir /vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labels_all_axis \
    # --resume_epoc 2 \
    # --img_name_filter /vol/aimspace/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/utils/mask_CT.csv \
    # --use_ablated_segmentations \
    # --eval_noshuffle_dataloader \
    # --eval_mask_removal \
    # --eval_blank_mask \
    # --eval_sample_size x \
