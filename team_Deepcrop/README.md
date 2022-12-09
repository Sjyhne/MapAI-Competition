### NORA MapAI: Precision 


Contact info: lilei@di.ku.dk
## For Task 1
we base [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for diverse model and data design

### install mmsegmentation 
MMSegmentation works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.3+.

    pip install -U openmim
    mim install mmcv-full
    cd mmsegmentation
    pip install -v -e .

### Run the evaluation
    step-1: download the model from google driver: 

https://drive.google.com/drive/folders/1jLslKQ62u1xGVP0VMaDgW8bBT-LTEHqW?usp=sharing
     to download "convext_base_aug18k_model.pth" to mmsegmentation folder

    step-2: change the data root to the test folder:
        '''
        open the file : /configs/_base_/datasets/map_building_aug.py
        change the line: data_root = '/home/dmn774/data/Deep1/benchmarks/NORA_MapAI/aug_data' to test_root'
    
    step-3: run the eval_test_building.sh

Note: 
#### for model structure:
we try ConvNeXt [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) , 
    Segformer [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203),
    Unet-structure, Deeplab-structure.
    
Currently, the ConvNext with data augmentation is best in our experiments.

#### for Data augmentation
we use other data resource about building segmentation to enhance the NORA MapAI data, with different data scales for training, which contains other 5000 and 18000 images. 
we also combine some augmentation, including RandomCrop, RandomFlip, PhotoMetricDistortion, Pad,Normalize.
#### for processing
1. we use 21k imagenet pretrained model based on the ConvNext model structure.

   (Because the building segmentation is binary segmentation, if more objects need to be segmented, the pretrained model can be efficiently used.)
2. we use rectangle aware post-processing for the predicted mask. 

## For Task2
we try to apply the mmsegmentation to adapt it, but the results is not good. We will also update the related codes.

now for the better performance with original codebase, go to the team_Deepcrop/src folder
    1. download the model from google driver:
        https://drive.google.com/drive/folders/1jLslKQ62u1xGVP0VMaDgW8bBT-LTEHqW?usp=sharing
    to download "best_task2.pt" to task2_submission_folder

    2. sh eval_task2.sh

## Team info
Team name: Deepcrop
Team participants: 
Emails: heyc2u@gmail.com
Country: Denmark