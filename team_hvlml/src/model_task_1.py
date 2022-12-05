import pathlib
import shutil

import cv2
from fastai.vision.all import *
from huggingface_hub import snapshot_download
from tqdm import tqdm

from competition_toolkit.dataloader import create_dataloader
from competition_toolkit.eval_functions import biou, iou


def main(args):

    #########################################################################
    ###
    # Load configuration
    ###
    #########################################################################
    with open(args.config, "r") as f:
        opts = yaml.load(f, Loader=yaml.Loader)
        opts = {**opts, **vars(args)}
    
    #########################################################################
    ###
    # Download model weights and load learner
    ###
    #########################################################################

    models_path = snapshot_download(repo_id="HVL-ML/MapAI-2022", 
                                    allow_patterns="*.pkl")

    model_list = glob.glob(models_path+'/task_1*.pkl')
    
    if len(model_list) == 1:
        ensemble = False
        model = model_list[0]
    elif len(model_list) > 1:
        ensemble=True
    else:
        raise Exception

    if not ensemble:
        if opts["device"]=="cpu":
            learn = load_learner(model, cpu=True)
        else:
            learn = load_learner(model, cpu=False)

    else:
        if opts["device"]=="cpu":
            learners = [load_learner(model_list[i], cpu=True) for i in range(len(model_list))]
        else:
            learners = [load_learner(model_list[i], cpu=False) for i in range(len(model_list))]
    #########################################################################
    ###
    # Create needed directories for data
    ###
    #########################################################################
    task_path = pathlib.Path(args.submission_path).joinpath(f"task_{opts['task']}")
    opts_file = task_path.joinpath("opts.yaml")
    predictions_path = task_path.joinpath("predictions")
    if task_path.exists():
        shutil.rmtree(task_path.absolute())
    predictions_path.mkdir(exist_ok=True, parents=True)

    #########################################################################
    ###
    # Load Data
    ###
    #########################################################################
    device = opts["device"]
    dataloader = create_dataloader(opts, opts["data_type"])
    print(dataloader)

    #########################################################################
    ###
    # Get predictions and scores
    ###
    #########################################################################

    iou_scores = np.zeros((len(dataloader)))
    biou_scores = np.zeros((len(dataloader)))

    for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference",
                                              leave=False):
        # Split filename and extension
        filename_base, file_extension = os.path.splitext(filename[0])

        # Load data
        data_type = opts["data_type"]

        img_fn = f"../../data/{data_type}/images/{filename[0]}"
        label_fn = f"../../data/{data_type}/masks/{filename[0]}"
        img = PILImage.create(img_fn)
        label = TensorMask(PILImage.create(label_fn))[:,:,0]

        label = np.uint8(label)

        # Model prediction
        if not ensemble:
            pred, _, probs = learn.predict(img)
            prediction = np.uint8(pred)
        else:
            all_probs = []
            i=1
            for learn in learners:
                print(f"Getting prediction from model #{i}/{len(learners)} ")
                _,_,probs = learn.predict(img)
                all_probs.append(probs)
                i+=1
            all_probs = torch.stack(all_probs, dim=0)
            mean = all_probs.mean(axis=0)
            pred = torch.where(mean>0.5, 0, 1)[0,::]
            prediction = np.uint8(pred)
    
        assert prediction.shape == label.shape, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {label.shape}]"

        # Predict score
        iou_score = iou(prediction, label)
        biou_score = biou(label, prediction)

        iou_scores[idx] = np.round(iou_score, 6)
        biou_scores[idx] = np.round(biou_score, 6)


        # Visualize
        #fig, ax = plt.subplots(1, 3)
        #columns = 3
        #rows = 1
        #ax[0].set_title("Input (RGB)")
        #ax[0].imshow(img)
        #ax[1].set_title("Prediction")
        #ax[1].imshow(prediction)
        #ax[2].set_title("Label")
        #ax[2].imshow(label)

        # Save to file.
        #predicted_sample_path_png = predictions_path.joinpath(f"{filename_base}.png")
        #predicted_sample_path_tif = predictions_path.joinpath(filename[0])
        #plt.savefig(str(predicted_sample_path_png))
        #plt.close()
        #cv2.imwrite(str(predicted_sample_path_tif), prediction)

        
    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    # Dump file configuration
    yaml.dump(opts, open(opts_file, "w"), Dumper=yaml.Dumper)
