#sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_20k_mapbuilding.py 1 \
	#        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding --deterministic

#sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_20k_mapbuilding.py 1 \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding  --deterministic

#sh tools/dist_train.sh configs/convnext/upernet_convnext_large_fp16_512x512_80k_mapbuilding.py 1 \
#	        --work-dir work_dirs/upernet_convnext_large_fp16_512x512_80k_mapbuilding  --deterministic

#python3 tools/train.py  configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding-1class.py \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_1class  --deterministic
##
#
#python3 tools/train.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding-1class.py \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_1class  --deterministic

#python3 tools/train.py  configs/segformer/segformer_mit-b0_512x512_160k_mapbuilding20k.py \
#	        --work-dir work_dirs/segformer_mit-b0_512x512_160k_mapbuilding20k  --deterministic
#python3 tools/train.py  configs/segformer/segformer_mit-b4_512x512_160k_mapbuilding20k.py \
#	        --work-dir work_dirs/segformer_mit-b4_512x512_160k_mapbuilding20k  --deterministic

#python3 tools/train.py  configs/convnext/upernet_convnext_base_fp16_1024x1024_80k_mapbuilding_aug.py \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_1024x1024_80k_mapbuilding_aug  --deterministic
##
#
#python3 tools/train.py  configs/convnext/upernet_convnext_xlarge_fp16_512x512_80k_mapbuilding_aug.py \
#	        --work-dir work_dirs/upernet_convnext_xlarge_fp16_512x512_80k_mapbuilding_aug  --deterministic


python3 tools/train.py  configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug_randomcrop256.py \
	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug_randomcrop256  --deterministic \
	        --load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600.pth
