epochs=${1:-"40"}
data_ratio=${2:-"0.4"}

cd src
echo "Training first model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai" --task 1 --backbone "timm-resnest26d" --batch-size 6 --image-size 1024

echo "Training Second Model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_edge" --task 1 --backbone "timm-resnest26d" --batch-size 6 --image-size 1024

echo "Training third model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai" --task 2 --backbone "timm-resnest26d" --batch-size 6 --image-size 1024

echo "Training fourth model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_edge" --task 2 --backbone "timm-resnest26d" --batch-size 6 --image-size 1024

echo "Training fifth model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_edge" --task 1 --backbone "efficientnet-b1" --batch-size 5 --image-size 1024

echo "Training sixth model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_edge" --task 2 --backbone "efficientnet-b1" --batch-size 5 --image-size 1024


