# https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
# https://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_one_hot(tensor,nClasses):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w, device=tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
        target_oneHot = to_one_hot(target, self.classes)
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)

        # Numerator Product
        #print("sizes:", inputs.size(), target_oneHot.size())
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = (inter/union)

        ## Return average loss over classes and batch
        return -loss.mean()