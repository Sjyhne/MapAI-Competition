# visulize the output of the model

from black import out
import matplotlib.pyplot as plt
import numpy as np
from augmentation.dataloader import create_dataloader
# from competition_toolkit.dataloader import create_dataloader
from ai_models.load_models import load_model
from ai_models.create_models import load_resnet50, load_resnet101, load_unet
from ai_models.utils import get_opts


def visualize_output(model, image, label, get_output):
    model.eval()

    output = model(image.cuda())[get_output]

    output = output.cpu().detach().numpy()
    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    image = np.transpose(image[0], (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))
    output = output.astype(np.uint8)
    output = np.transpose(output[0], (1, 2, 0))
    output = 1 - output.argmax(axis=2)

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(label)
    ax[1].set_title("Label")
    ax[2].imshow(output)
    ax[2].set_title("Output")
    plt.show()

# This visualizes the model training and test output
def visulize_training(path_training_folder):
    # get run log file
    # {'epoch': 0, 'trainloss': 0.1063, 'testloss': 0.0795, 'trainiou': 0.5633, 'testiou': 0.5732, 'trainbiou': 0.4875, 'testbiou': 0.5, 'trainscore': 0.5254, 'testscore': 0.5366}
    # get trainscore and testscore
    # plot trainscore and testscore

    train_scores = []
    test_scores = []
    with open(path_training_folder + "/run.log", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = line.split(" ")
                train_scores.append(float(line[15].replace(",", "")))
                test_scores.append(float(line[17].replace(",", "").replace("}", "")))
    
    plt.plot(train_scores, label="train")
    plt.plot(test_scores, label="test")
    plt.legend()
    plt.savefig(path_training_folder + "/train_test_score.png")
    plt.show()
    # save
    plt.close()
    

    

if __name__ == "__main__":
    opts = get_opts()
    model_path = "backup\\run_26\\best_task1_7.pt"
    # model_path = "backup\\resnet50_scratch\\best_task1_4.pt"
    model, get_output = load_model(load_unet, model_path, opts)
    dataloader = create_dataloader(opts, "train")
    for idx, batch in enumerate(dataloader):
        image, label, filename = batch
        visualize_output(model, image, label, get_output)

    run_log_path = "backup\\unet_with_augmentation"
    model_path = "pretrained_task2.pt"
    visulize_training(run_log_path)

            
    