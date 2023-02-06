#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 --mem=32G
#SBATCH --job-name="seg"
#SBATCH -p gpu --gres=gpu:1 -x hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl
#SBATCH --time=4-15:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilei@di.ku.dk
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
#sh seg_sh/sb_train_Min_denmark_pl_2205_sampling1.sh
#python train.py --task 2 --name combine_lidar
python train.py --task 2 --name deeplabv3_resnet50_withpreweight_combine_lidar --data_ratio 1.0
#python train.py --task 1 --name image_resnet101_withweight
#python train.py --task 1 --name  deeplabv3_mobilenet_v3_large
#python train.py --task 2 --name  deeplabv3_mobilenet_v3_large_withpreweight_combine_lidar --config config/data_deeplab3.yaml --data_ratio 1.0

#python train.py --task 2 --name  simple_resNetUnet_task2
#python train_UNetFormer.py --task 1 --name  FTUNetFormer --data_ratio 1.0 --lr 1e-4
#python train_UNetFormer.py --task 1 --name  UNetFormer --data_ratio 1.0
#python train_FTUNetFormer.py --task 1 --name  FTUNetFormer_default --data_ratio 1.0 --lr 1e-2 --wandb run
#python train.py --task 2 --name  deeplabv3_resnet50_combine_lidar --config config/data_deeplab3.yaml --data_ratio 1.0
#python train_FTUNetFormer.py --task 1 --name  FTUNetFormer_defaultTUNetFormer_default_pretrain_lr6e-2 --data_ratio 1.0 --lr 6e-2 --wandb run

#echo "try for using the tverskyloss"s
#python train_withSOTA.py --task 1 --name deeplabv3_resnet50_with_pretrain_tverskyloss --data_ratio 1.0
#python train_withSOTA.py --task 1 --name UnetPlusPlus_resnet34 --data_ratio 1.0

#python train_segformer.py --task 1 --name segformer_croloss --data_ratio 1.0 --config config/data_segformer.yaml
#python train_ConvNet.py --task 1 --name segformer_croloss --data_ratio 0.01 --config config/data_segformer.yaml
#python train_ConvNet.py --task 2 --name conv_croloss_withlidar --data_ratio 0.01 --config config/data_segformer.yaml

