sh tools/dist_train.sh configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py 1 \
	        --work-dir work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_21kPretrained --deterministic \
		#        --options model.pretrained="pretrained_model/convnext-base_3rdparty_in21k_20220301-262fd037.pth"
        --options model.backdone.init_cfg.Pretrained="pretrained_model/convnext-base_3rdparty_in21k_20220301-262fd037.pth"
