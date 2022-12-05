
import pathlib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import gdown
import os
import shutil
import glob
from competition_toolkit.dataloader import create_dataloader
from competition_toolkit.eval_functions import iou, biou

import pytorch_lightning as pl

from utils import SimpleBuildingLidarDataset, LidarNet
from competition_toolkit.eval_functions import iou, biou
from PIL import Image
import numpy as np

def get_sorted_data_paths(split):
    image_paths = sorted(glob.glob(split + "/images/*.tif"))
    lidar_paths = sorted(glob.glob(split + "/lidar/*.tif"))
    masks_paths = sorted(glob.glob(split + "/masks/*.tif"))

    return image_paths, lidar_paths, masks_paths

def main(args):

    submission_path = args.submission_path
    data_type = args.data_type

    torch.cuda.is_available()

    print("Getting the weighths")

    pt_share_link = "https://drive.google.com/file/d/15lsdB8ZkLaNOTURp3tS6EcZTdyKK8j-t/view?usp=sharing"
    pt_id = pt_share_link.split("/")[-2]

    # Download trained model ready for inference
    url_to_drive = f"https://drive.google.com/uc?id={pt_id}"
    model_checkpoint = "pretrained_task2.ckpt"

    gdown.download(url_to_drive, model_checkpoint, quiet=False)

    model =  LidarNet("unetplusplus", "vgg19", in_channels=1, out_classes=1)

    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    image_paths_test, lidar_paths_test, masks_paths_test = get_sorted_data_paths(data_type)

    test_dataset = SimpleBuildingLidarDataset(lidar_paths_test, masks_paths_test)

    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=n_cpu)
    
    iou_scores = np.zeros((len(test_dataloader)))
    biou_scores = np.zeros((len(test_dataloader)))
    
    idx = 0
    for batch in test_dataloader:

        gt_mask = batch["mask"]
        with torch.no_grad():
            model.eval()
            logits = model(batch["lidar"])
        pr_masks = logits.sigmoid()
        
        prediction = np.uint8(pr_masks.numpy().squeeze())
        label = np.uint8(gt_mask.numpy().squeeze())

        assert prediction.shape == label.shape, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {label.shape}]"

                # Predict score
        iou_score = iou(prediction, label)
        biou_score = biou(label, prediction)

        print("Curr image metrics", iou_score, biou_score)

        iou_scores[idx] = np.round(iou_score, 6)
        biou_scores[idx] = np.round(biou_score, 6)

        prediction_visual = np.copy(prediction)

        lidar = batch["lidar"].numpy().squeeze(axis=0).transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 3)
        columns = 3
        rows = 1
        ax[0].set_title("Input (LIDAR)")
        ax[0].imshow(lidar)
        ax[1].set_title("Prediction")
        ax[1].imshow(prediction_visual)
        ax[2].set_title("Label")
        ax[2].imshow(label)

        # Save to file.
        isExist = os.path.exists(submission_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(submission_path)

        predicted_sample_path_png = submission_path + str(idx) + "lidar.png" 
        #plt.savefig(predicted_sample_path_png)

        mask_path = batch["path"][0][-15:]
        submission_img = mask_path.replace("/","")
        # Saving the image
        #img.save(submission_path + submission_img)
        import cv2
        full_path = submission_path +"/"+ submission_img
        cv2.imwrite(full_path, prediction_visual)
        label = cv2.imread(full_path, cv.IMREAD_GRAYSCALE)
        label[label == 255] = 1
        label = cv.resize(label, (500,500))
        cv2.imwrite(full_path, label)
        plt.close()

        idx += 1

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

if __name__ == "__main__":
    
    main()

