
echo "forward to get the prediction"
python tools/test.py configs/convnext/upernet_convnext_base_fp16_512x512_80k_mapbuilding.py \
        convext_base_aug18k_model.pth  \
					--show-dir show_dirs/convext_base_aug18k_model

echo "run the eval function"

python tools/eval_iou_biou.py --submission-path show_dirs/convext_base_aug18k_model --team Deepcrop \
        --task 1 --data-type validation \
        --gt-root /home/dmn774/data/Deep1/benchmarks/NORA_MapAI/data/validation/masks/

echo " please conform the gt-root is the test mask path"
echo " validation dataset metrics: avg_iou 0.94432 avg_biou 0.88187"
#					convext_base_aug18k_model.pth \
#          convext_base_aug18k_model.pth  \

## work_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k/iter_41600.pth  \
## --show-dir show_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug18k/iter_lastest_mask/
##show_dirs/upernet_convnext_base_fp16_512x512_80k_mapbuilding_finetuneFrom80k_aug5000/iter_41600_mask --team Deepcrop