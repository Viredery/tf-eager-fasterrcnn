import numpy as np

class DataGenerator(object):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __call__(self):
        indices = np.arange(len(self.dataset))
        for img_idx in indices:
            img, img_meta, bbox, label = self.dataset[img_idx]
            yield img, img_meta, bbox, label
