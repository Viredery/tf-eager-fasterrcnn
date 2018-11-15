import cv2
import numpy as np


def img_flip(img):
    '''Flip the image horizontally
    
    Args
    ---
        img: [height, width, channel]
    '''
    return np.fliplr(img)

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
    '''Normalize the image.
    
    Args
    ---
        img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    '''
    img = img.astype(np.float32)
    return (img - mean) / std    

def imdenormalize(norm_img, mean, std):
    '''Denormalize the image.
    
    Args
    ---
        norm_img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    '''
    return norm_img * std + mean


def generate_ori_img(img, img_meta, mean=(0, 0, 0), std=(1, 1, 1)):
    '''Recover the origanal image.
    
    Args
    ---
        img: np.ndarray. [height, width, channel]. The transformed image.
        img_meta: np.ndarray. [11]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    '''
    ori_shape = img_meta[0:2].astype(np.int32)
    img_shape = img_meta[3:5].astype(np.int32)
    flip = img_meta[10].astype(np.int32)
    
    img = img[:img_shape[0], :img_shape[1]]
    if flip:
        img = img_flip(img)
    img = cv2.resize(img, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    img = imdenormalize(img, mean, std)
    return img