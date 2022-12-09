
mkdir -p ../data/mapai/train/masks_reclassified
cd src/data

echo "Reclassifying masks"
python reclassify_masks.py
