# Team_Fundator MapAI Submission

This repository contains all files related to our submission for the MapAI Building segmentation competition.

Our solution is based on UNet ensembles with heterogneity from differing backbones and training data.

An archive with pretrained models is available in [Google Drive](https://drive.google.com/drive/folders/1SQnS-cczKYae0_FpBFchFGpZ3X4QFyZo?usp=share_link)

## Requirements

### Software
The code has been tested on Ubuntu 20.04 LTS and Windows 10 with CUDA 11.7 and python 3.8.

We provide .sh scripts (tested on Ubuntu 20.04 LTS) to replicate the training for our final submission.

### Hardware
Our implementation was trained using an RTX 3090 GPU with 24 GB of VRAM. If you have less than 8GB available consider lowering the imagesize to 512 in the config, and use a batch size no greater than 4.

Our data preparation consumes ~100GB of disk space.

## Installation

In the competitions root folder install the competition toolkit and our dependencies:

    pip install competition_toolkit/ team_fundator/

## Data preparation
We have made a script, `prepare_data.sh`, to complete our data-preprocessing steps:
* Download the dataset if it isn't already.
* Stitch the data into the original 10000 * 5000 and 5000 * 5000 images.
* Split the images with a stride of 500/3

To run the script, simply write:

    sh prepare_data.sh

The script `reclassify_data.sh` reclassifies the masks with building edges and regions in-between two adjacent buildlings as separate classes.

## Training

From the data preparation you should have the following data structure:
    
    MapAI-Competition
    |
    └─── data 
    |   └───  big_tiles
    |   |   └─── ...
    |   └─── mapai
    |   |   └─── train
    |   |       | images/
    |   |       | lidar/
    |   |       | masks/
    |   |       | mask_reclassified/
    |   | train
    |   | validation
    └─── team_fundator
    │   │   ...
    |   └───
To train models run `train_task*.sh <epochs> <data_ratio>`. For our standard parameters with 50 epochs and 0.4 data ratio, no arguments are needed:

    sh train_task1.sh
    sh train_task2.sh
    
Lastly, train two models each for task 1 and 2 with 1024 image resolution on the `mapai` and `mapai_edge` datasets:  
    
    sh train_1024res.sh
    
    
    
## Description
The ensembles for task 1 and 2 are desdcribed here.
### Task 1
We selected five models for the task 1 ensemble, out of eight models which were trained with different combinations of encoders, datasets and imagesizes. The first 6 combinations are given by the Cartesian product of the following sets:
    
    Encoders: (timm-resnest-26d, efficientnet-b1)
    Datasets: (mapai, mapai_reclassified, mapai_edge)

The last two models were trained with image size 1024 using the  `timm-resnest26d` backbone with the  `mapai` and  `mapai_edge` datasets. 

See the section *Model Selection* for how we chose which models to include in the task 1 and 2 ensembles.

### Task 2
The ensemble contains three models out of ten models which were trained with different combinations of encoders, datasets and image sizes. The first 8 combinations are given by the Cartesian product of the following sets:
    
    Encoders: (timm-resnest-26d, efficientnet-b1)
    Datasets: (mapai, mapai_reclassified, mapai_lidar_masks, mapai_edge)


The dataset `mapai_lidar_masks` has a third class for the case where the LIDAR height is 0. `Mapai_edge` is similar to `mapai_reclassified`, but only has the additional edge class.

The last two models were trained with image size 1024 using the  `timm-resnest26d` backbone with the  `mapai` and  `mapai_edge` datasets.

### Model Selection
To select models for the ensemble, we first use `save_ensemble_preds.py` to save the concatenated predictions of each model for each task. Next we use `test_subensembles.py` to go through all combinations of 3 or more models from the concatenated predictions, using an even weighting of each prediction.

We then select the best performing subset of models for our ensemble, as well as the next best performing selection with at least one more model. Finally, we run the evolutionary algorithm in `bio_ensemble.py` to evolve optimized weights for the models selected for the ensemble, and choose the best performing weighting.
