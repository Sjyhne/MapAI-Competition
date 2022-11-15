epochs=${1:-"50"}
data_ratio=${2:-"0.4"}

cd src
echo "Training first model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_reclassified" --task 1 --backbone "timm-resnest26d"

echo "Training second model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai_reclassified" --task 1 --backbone "efficientnet-b1"

echo "Training third model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai" --task 1 --backbone "timm-resnest26d"

echo "Training fourth model"
python train.py --epochs $epochs --data-ratio $data_ratio --dataset "mapai" --task 1 --backbone "efficientnet-b1"




