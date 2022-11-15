cd src

mkdir -p ../data/mapai

echo "Downloading dataset"
python download.py

cd data
echo "Stitching dataset into original tiles"
python combine_data.py

echo "Splitting dataset with stride = 500/3"
python split_data.py



