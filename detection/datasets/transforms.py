import cv2
import numpy as np

class ImageTransform(object):
    '''
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


def img_flip(img):
    return np.flip(img, axis=1)

def bbox_flip(bboxes, img_shape):
    '''Flip bboxes horizontally.
    
    Args
    ---
        bboxes: [..., 4]
        img_shape: Tuple. (height, width)
    '''
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 1] = w - bboxes[..., 3] - 1
    flipped[..., 3] = w - bboxes[..., 1] - 1
    return flipped


def impad_to_square(img, pad_size):
    '''Pad an image to ensure each edge to equal to pad_size.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded
        pad_size: Int.
    
    Returns
    ---
        ndarray: The padded image with shape of [pad_size, pad_size, channels].
    '''
    shape = (pad_size, pad_size, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad

def impad_to_multiple(img, divisor):
    '''Pad an image to ensure each edge to be multiple to some number.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded.
        divisor: Int. Padded image edges will be multiple to divisor.
    
    Returns
    ---
        ndarray: The padded image.
    '''
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    
    
    shape = (pad_h, pad_w, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad

def imrescale(img, scale):
    '''Resize image while keeping the aspect ratio.
    
    Args
    ---
        img: [height, width, channels]. The input image.
        scale: Tuple of 2 integers. the image will be rescaled 
            as large as possible within the scale
    '''
    
    h, w = img.shape[:2]
    
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    
    new_size = (int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5))

    rescaled_img = cv2.resize(
        img, new_size, interpolation=cv2.INTER_LINEAR)
    
    return rescaled_img, scale_factor

def imnormalize(img, mean, std):
    img = img.astype(np.float32)
    return (img - mean) / std    
    