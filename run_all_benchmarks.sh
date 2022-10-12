#!/bin/bash


# I'm not sure why, wandb sometimes will quietly not sync some runs.
# Run offline and sync everythign at the end.

WANDB_MODE="offline"
WANDB_PROJECT="ready-steady-go"

MODELS="resnet50 swin_s3_tiny_224"
BATCHES="8 8 8 16 16 16 32 32 32" # 64 128 256 512 1024"

N_SECONDS=3


#set -x

wandb login

echo "Warming up the GPU for 3 minutes..."
# gpu-sprint --model=resnet50 --n_seconds=180

echo "Running benchmarks..."

# You can do multiple runs, but in my experience the results barely change between runs.
for RUN in 1 #2 3
do
    for m in $MODELS; do
        for fp16 in " " "--fp16"; do
            for bs in $BATCHES; do
                ready-steady-go --model=$m $fp16 --bs=$bs --n_seconds=$N_SECONDS --wnb=$WANDB_MODE --wnb_project=$WANDB_PROJECT --run_number=$RUN
                if [ $? -ne 0 ]; then
                    break # We probably hit a batch size the GPU can't handle
                fi
            done
        done
    done
done

wandb sync --sync-all