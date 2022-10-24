from importlib.metadata import requires
from tkinter import Variable
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import torch.functional as F
from torch.autograd import Variable

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

class BuildingClfDataset(Dataset):

    def __init__(self, root_dir, ext=".tif"):
        """
        Args:
            root_dir (string): Directory with all the images and .npy files.
        """
        self.data = glob.glob(root_dir + "outputs/*.npy")
        self.mask = glob.glob(root_dir + "masks/*" + ext)

    def __len__(self):
        return len(self.model_outputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data.iloc[idx, 1:]
        image = torch.tensor(np.load(image_name, dtype=np.float32))

        mask_name = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        mask = torch.tensor(cv2.imread(mask_name), dtype=torch.float32)
        sample = {'image': image, 'mask': mask}

        return sample

class SuperEnsemble(nn.Module):

    def __init__(self, ensemble_size, image_size) -> None:
        super().__init__()

        shape = image_size + (1, ensemble_size)
        init_w = np.random.normal(size=shape)

        self.W = Variable(init_w, requires_grad=True, dtype=torch.float32)
        self.b = Variable(torch.zeros(image_size), requires_grad=True, dtype=torch.float32)
        self.optimizer = torch.optim.Adam([self.W, self.b])

    def forward(self, features):
        return torch.squeeze(torch.matmul(self.W, features)) + self.b

def train_model(model, dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        model.optimizer.zero_grad()
        pred = model(sample_batched["image"])
        loss = dice_loss(pred, sample_batched["mask"])
        loss.backward()
        model.optimizer.step()

if __name__ == "__main__":

    ensemble_size = 3
    input_size = (2, 2)

    clf = SuperEnsemble(ensemble_size, input_size)
    data = BuildingClfDataset(root_dir="clf_outputs/")
    dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=8)
    train_model(clf, dataloader)