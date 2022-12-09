#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=2
#SBATCH  --cpus-per-task=4
# --mem=32G
#SBATCH --job-name="2-segformer"
## if using gpu: --gres=gpu:1
### ml4good
#SBATCH -p gpu --gres=gpu:a100:2 -x hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl
#SBATCH --time=5-15:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilei@di.ku.dk
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
python3 tools/train.py  configs/segformer/segformer_mit-b5_512x512_160k_mapbuilding20k.py \
	        --work-dir work_dirs/segformer_mit-b5_512x512_160k_mapbuilding20k_2gpus  --deterministic
