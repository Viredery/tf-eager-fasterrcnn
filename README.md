# tf-eager-fasterrcnn

Faster R-CNN R-101-FPN model was implemented with TensorFlow2.0 Eager Execution. 

# Requirements

- Cuda 10.0
- Python 3.5
- TensorFlow 2.0.0
- cv2

# Usage

see `train_model.ipynb`, `inspect_model.ipynb` and `eval_model.ipynb`


### Download trained Faster R-CNN

- [百度网盘](https://pan.baidu.com/s/1I5PGkpvnDSduJnngoWuktQ)


# Updating

- [ ] online hard examples mining
- [ ] soft-nms
- [ ] TTA (multi-scale testing, flip and box voting)
- [ ] multi-scale training

# Acknowledgement

This work builds on many excellent works, which include:

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
