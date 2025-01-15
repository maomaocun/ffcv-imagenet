# 8 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn50_configs/rn50_16_epochs.yaml \
    --data.train_dataset=/root/autodl-tmp/imagenet_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/root/autodl-tmp/imagenet_ffcv/val_500_0.50_90.ffcv \
    --data.num_workers=32 --data.in_memory=1 \
    --logging.folder=./