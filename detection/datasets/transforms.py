import numpy as np

from detection.datasets.utils import *

class ImageTransform(object):
    '''Preprocess the image.
    
        1. rescale the image to expected size
        2. normalize the image
        3. flip the image (if needed)
        4. pad the image (if needed)
    '''
    def __init__(self,
                 scale=(800, 1333),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed'):
        self.scale = scale
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pad_mode = pad_mode
            
        self.impad_size = max(scale) if pad_mode == 'fixed' else 64

    
    def __call__(self, img, flip=False):
        img, scale_factor = imrescale(img, self.scale)
        img_shape = img.shape
        img = imnormalize(img, self.mean, self.std)
          
        if flip:
            img = img_flip(img)
        if self.pad_mode == 'fixed':
            img = impad_to_square(img, self.impad_size)
        else: # 'non-fixed'
            img = impad_to_multiple(img, self.impad_size)
        
        return img, img_shape, scale_factor

class BboxTransform(object):
    '''Preprocess gt bboxes.
    
        1. rescale bboxes according to image size
        2. flip bboxes (if needed)
    '''
    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts
    
    def __call__(self, bboxes, labels, 
                 img_shape, scale_factor, flip=False):
        
        if self.max_num_gts is not None and bboxes.shape[0] > self.max_num_gts:
            concated = np.concatenate([bboxes, np.expand_dims(labels, 1)], axis=1)
            np.random.shuffle(concated)
            concated = concated[:self.max_num_gts]
            
            bboxes, labels = np.split(concated, [4], axis=1)
            labels = np.squeeze(labels, 1)

        
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
            
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[0])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[1])
            
        return gt_bboxes, labels.astype(np.int32)
