from kornia.losses import FocalLoss, TverskyLoss
from kornia.losses import focal_loss, tversky_loss, lovasz_hinge_loss, lovasz_softmax_loss
from fastai.losses import CrossEntropyLossFlat, DiceLoss, FocalLossFlat
from fastai.vision.all import *


class CombinedCEDiceLoss:
    "CE and Dice combined"
    def __init__(self, axis=1, smooth=1e-06, eta=1.):
        #store_attr()
        self.axis=axis
        self.smooth = smooth
        self.eta = eta
        self.cross_entropy = CrossEntropyLossFlat(axis=axis)
        self.dice_loss =  DiceLoss(axis, smooth=smooth, reduction='mean')
        
    def __call__(self, pred, targ):
        return self.cross_entropy(pred, targ) + self.eta * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


class CombinedFocalDiceLoss:
    "Focal and Dice combined"
    def __init__(self, axis=1, gamma=2., eta=1.):
        #store_attr()
        self.axis=axis
        self.gamma = gamma
        self.eta = eta
        self.focal_loss = FocalLossFlat(axis=axis, gamma=gamma)
        self.dice_loss = DiceLoss(axis=axis, reduction='mean')

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.eta * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


class CombinedFocalTanimotoLoss:
    "Focal and Tanimoto combined"
    
    def __init__(self, axis=1, smooth=1., alpha=1., beta=1., gamma=2., eta=1.):
        #store_attr()
        self.axis=axis
        self.alpha = alpha # alpha=beta=1 in Tversky is Tanimoto
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.focal_loss = FocalLossFlat(axis=axis, gamma=gamma)
        self.tversky_loss =  TverskyLoss(alpha=alpha, beta=beta)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.eta*self.tversky_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class CombinedCETverskyLoss:
    "CE and Tversky combined"
    
    def __init__(self, axis=1, smooth=1., beta=0.8, gamma=0.25, eta=0.5):
        #store_attr()
        self.axis=axis
        alpha = 1 - beta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.cross_entropy = CrossEntropyLossFlat(axis=axis)
        self.tversky_loss =  TverskyLoss(alpha=self.alpha, beta=beta)
        
    def __call__(self, pred, targ):
        return self.cross_entropy(pred, targ) + self.eta * self.tversky_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


class CombinedCEFocalTverskyLoss:
    "CE and TverskyFocal combined"
    
    def __init__(self, axis=1, beta=0.8, gamma=0.25, eta=0.5):
        #store_attr()
        self.axis=axis
        alpha = 1 - beta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.ce_loss = CrossEntropyLossFlat(axis=axis)
        self.tversky_loss =  TverskyLoss(alpha=self.alpha, beta=self.beta)
        
    def __call__(self, pred, targ):
        ce_loss = self.ce_loss(pred, targ)
        focaltversky_loss = (1-self.tversky_loss(pred, targ))**self.gamma
        return ce_loss + self.eta * focaltversky_loss
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


### Multispectral losses ###
# _Comment about this TBA_

def multispectral_ce_dice_loss(pred, targ):
    axis=1
    alpha = 1.
    
    cross_entropy = CrossEntropyLossFlat(axis=axis)
    dice_loss =  DiceLoss(axis=axis, reduction='mean')
    
    return cross_entropy(pred, targ) + alpha * dice_loss(pred, targ)

def multispectral_focal_dice_loss(pred, targ):
    axis=1
    eta = 1.
    gamma = 2.0
    
    focal_loss = FocalLossFlat(axis=axis, gamma=gamma)
    dice_loss =  DiceLoss(axis=axis, reduction='mean')
    
    return focal_loss(pred, targ) + eta * dice_loss(pred, targ)


def multispectral_ce_tversky_loss(pred, targ):
    axis=1
    beta = 0.8 # FN
    alpha = 1-beta # FP. Note: Tversky equals F beta coefficient since alpha + beta = 1
    eta = 0.5
    ce_loss = CrossEntropyLossFlat(axis=axis)
    t_loss =  tversky_loss(pred, targ, alpha=alpha, beta=beta)
    
    return ce_loss(pred, targ) + eta*t_loss


def multispectral_focal_tversky_loss(pred, targ):
    axis=1
    beta = 0.8 # FN
    alpha = 1-beta # FP. Note: Tversky equals F beta coefficient since alpha + beta = 1
    gamma = 2.0
    eta = 1.
    focal_loss = FocalLossFlat(axis=axis, gamma=gamma)
    t_loss =  tversky_loss(pred, targ, alpha=alpha, beta=beta)
    
    return focal_loss(pred, targ) + eta*t_loss


def multispectral_ce_focaltversky(pred, targ):
    axis=1
    gamma = 0.25
    beta = 0.8 # FN
    alpha = 1-beta # FP 
    eta = 0.5
    ce_loss = CrossEntropyLossFlat(axis=axis)
    t_loss =  tversky_loss(pred, targ, alpha=alpha, beta=beta)
    
    return ce_loss(pred, targ) + eta*(1-t_loss)**gamma


def multispectral_focal_tanimoto_loss(pred, targ):
    axis=1
    gamma = 2.
    alpha = 1. #Tanimoto alpha=beta=1 in Tversky
    beta = 1.  #Tanimoto

    f_loss = focal_loss(pred, targ, alpha=alpha, gamma=gamma, reduction='mean')
    t_loss = tversky_loss(pred, targ, alpha=alpha, beta=beta) 

    return f_loss + t_loss


def multispectral_ce_tversky_lovaszhinge_loss(pred, targ):
    axis=1
    gamma = 0.25
    beta = 0.8 # FN
    alpha = 1-beta # FP. Note: Tversky equals F beta coefficient since alpha + beta = 1
    eta = 0.5
    zeta = 0.5
    ce_loss = CrossEntropyLossFlat(axis=axis)
    t_loss =  tversky_loss(pred, targ, alpha=alpha, beta=beta)
    lh_loss = lovasz_softmax_loss(pred, targ)
    
    return ce_loss(pred, targ) + eta*(1-t_loss)**gamma + zeta*lh_loss