
#python tools/test.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py \
#	        work_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding/iter_20000.pth  \
#	        --show-dir show_dirs/upernet_convnext_base_fp16_512x512_20k_mapbuilding_mask/ \
#	        --eval mIoU

#	        work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600.pth  \

#python tools/test.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py \
#	         work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/latest.pth  \
#	        --show-dir show_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_lastest_mask/

python tools/test.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py \
	        work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k/latest.pth  \
	        --show-dir show_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k/iter_lastest_mask/