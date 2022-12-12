import albumentations as A
from fastai.vision.all import *
from multispectral import *

# Multispectral #

class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx=0 #only training
    def __init__(self, aug, **kwargs): 
        super().__init__(**kwargs)
        self.aug = aug
        
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img.permute(1,2,0)), mask=np.array(mask))
        return TensorImage(aug['image'].transpose(2,0,1)), TensorMask(aug['mask'])

    
    
def get_multiband_item_tfms(size:(list, tuple)=(500,500), p=0.5): 
    item_tfms = SegmentationAlbumentationsTransform(A.Compose([
                                                     A.RandomCrop(size[0], size[1]), 
                                                     A.HorizontalFlip(p=p), 
                                                     A.VerticalFlip(p=p),
                                                     A.RandomBrightnessContrast(p=p), 
                                                     A.RandomRotate90(p=p), 
                                                     A.ShiftScaleRotate(p=p)
                                                    ])) 
    
    return item_tfms




def get_multiband_dataloaders(df, x_name, y_name, codes, splitter, bs, 
                    item_tfms, batch_tfms):

    dblock = DataBlock(
                blocks=(ImageBlock(MImage), MaskBlock(codes=codes)),
                splitter=splitter,
                get_x=ColReader(x_name),
                get_y=ColReader(y_name),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
    
    dls = dblock.dataloaders(df, bs=bs)
    return dls
