import albumentations as A

def get_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.75,),
        # A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    # tf$image$random_brightness(max_delta = 0.3) %>% 
    # tf$image$random_contrast(lower = 0.5, upper = 0.7) %>% 
    # tf$image$random_saturation(lower = 0.5, upper = 0.7) %>% album.RandomCrop(height=256, width=256, always_apply=True),
      
        
