import torch 
import PIL 
import matplotlib
from fastai.data.all import *


def custom_load_image(fn,  cls=torch.Tensor):
    image_fn, lidar_fn = fn.split(';')

    aerial_img = PIL.Image.open(image_fn)
    lidar_img =  PIL.Image.open(lidar_fn)

    aerial_tensor = torch.Tensor(np.array(aerial_img))
    aerial_tensor = aerial_tensor.permute(2,0,1)
    lidar_tensor = torch.Tensor(np.array(lidar_img))[None]
    
    t =  torch.cat((aerial_tensor, lidar_tensor), 0)    
    
    return cls(t)

class MImage(TensorImage):
    
    def __init__(self, x, chnls_first=False):
        self.chnls_first = chnls_first

    @classmethod
    def create(cls, fn:(Path,str,ndarray)):
        
        if isinstance(fn, (Path,str)): fn = custom_load_image(fn=fn, cls=torch.Tensor)
        elif isinstance(fn, ndarray): fn = torch.from_numpy(fn)
        
        return cls(fn)
    
    def __repr__(self):
        
        return (f'MImage: {self.shape}')