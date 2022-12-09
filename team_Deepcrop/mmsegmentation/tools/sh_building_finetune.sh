#sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_20k_mapbuilding.py 1 \
	#        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding --deterministic

#sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py 1 \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom20k  --deterministic \
#	        --load-from work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding/latest.pth

#sh tools/dist_train.sh configs/convnext/upernet_convnext_large_fp16_512x512_80k_mapbuilding.py 1 \
#	        --work-dir work_dirs/upernet_convnext_large_fp16_512x512_80k_mapbuilding  --deterministic


#python tools/train.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug.py  \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000  --deterministic \
#	        --load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom20k/latest.pth


#python tools/train.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug18k.py  \
#	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k_finetuneFrom20k  --deterministic \
#	        --load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom20k/latest.pth
#
#
#python tools/train.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug18k.py  \
#--work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k_finetunefrom8k_aug5000_finetuned  --deterministic \
#--load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600.pth

#python tools/train.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug18k.py  \
#--work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k_finetunefrom8k_aug18_finetuned  --deterministic \
#--load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k_finetuneFrom20k/latest.pth

python3 tools/train.py  configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug_randomcrop128.py \
	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_aug_randomcrop480  --deterministic \
	        --load-from work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600.pth