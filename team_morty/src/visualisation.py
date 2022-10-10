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
    output = output.argmax(axis=2)

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(label)
    ax[1].set_title("Label")
    ax[2].imshow(output)
    ax[2].set_title("Output")
    plt.show()
    

if __name__ == "__main__":
    opts = get_opts()
    # model_path = "backup\\resnet50_scratch\\best_task1_4.pt"
    # model_path =  "runs\\task_1\\run_114\\best_task1_0.pt"
    model_path = "run_8\\run_8\\best_task1_7.pt"
    # model_path = "runs\\task_1\\run_88\\best_task1_0.pt"
    # model_path = "backup\\unet_with_augmentation\\best_task1_8.pt"
    model, get_output = load_model(load_resnet101, model_path, opts)
    # model, get_output = load_model(load_resnet50, model_path, opts)
    dataloader = create_dataloader(opts, "train")
    for idx, batch in enumerate(dataloader):
        image, label, filename = batch
        visualize_output(model, image, label, get_output)
            
    