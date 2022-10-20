#!/bin/bash

WANDB_MODE="online"
WANDB_PROJECT="ready-steady-go"

MODELS="resnet50 vgg19 swin_s3_base_224"
BATCHES="8 16 32 64 128 256 512 1024 2048 4096"

N_SECONDS=30

#set -x

nvidia-smi -q > gpu-info.txt
cat /proc/cpuinfo > cpu-info.txt

wandb login

echo "Warming up the GPU for 3 minutes..."
ready-steady-go --model=resnet50 --n_seconds=180

echo "Running benchmarks..."

# You can do multiple runs, but in my experience the results barely change between runs.
for RUN in 1 #2 3
do
    for m in $MODELS; do
        for fp16 in " " "--fp16"; do
            for bs in $BATCHES; do
                ready-steady-go --model=$m $fp16 --bs=$bs --n_seconds=$N_SECONDS \
                    --wnb=$WANDB_MODE --wnb_project=$WANDB_PROJECT --run_number=$RUN
                if [ $? -ne 0 ]; then
                    # We probably hit a batch size the GPU can't handle.
                    # No need to try larger batch sizes.
                    break
                fi
            done
        done
    done
done

# Sync everything just in case. On a rare occasion wandb forgets to update symmary otherwise.
wandb sync --sync-all --include-synced
