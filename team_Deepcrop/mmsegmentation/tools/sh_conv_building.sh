#sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_20k_mapbuilding.py 1 \
#        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding --deterministic

sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_40k_mapbuilding.py 1 \
        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_40k_mapbuilding --deterministic
