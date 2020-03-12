import numpy as np
import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import *

class ProposalTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 num_classes=81):
        '''Compute regression and classification targets for proposals.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            num_rcnn_deltas: int. Maximal number of RoIs per image to feed to bbox heads.

        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_classes = num_classes
            
    def build_targets(self, proposals, gt_boxes, gt_class_ids, img_metas):
        '''Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.
        
        Args
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]
            
        Returns
        ---
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs: [batch_size * num_rois]. Integer class IDs.
            rcnn_target_deltas: [batch_size * num_rois, (dy, dx, log(dh), log(dw))].
            
        '''
        
        pad_shapes = calc_pad_shapes(img_metas)
        batch_size = img_metas.shape[0]
        
        proposals = tf.reshape(proposals[:, :5], (batch_size, -1, 5))
        
        rcnn_rois = []
        rcnn_labels = []
        rcnn_label_weights = []
        rcnn_delta_targets = []
        rcnn_delta_weights = []

        
        for i in range(batch_size):
            rois, labels, label_weights, delta_targets, delta_weights = self._build_single_target(
                proposals[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i], i)
            rcnn_rois.append(rois)
            rcnn_labels.append(labels)
            rcnn_label_weights.append(label_weights)
            rcnn_delta_targets.append(delta_targets)
            rcnn_delta_weights.append(delta_weights)

        rcnn_rois = tf.concat(rcnn_rois, axis=0)
        rcnn_labels = tf.concat(rcnn_labels, axis=0)
        rcnn_label_weights = tf.concat(rcnn_label_weights, axis=0)
        rcnn_delta_targets = tf.concat(rcnn_delta_targets, axis=0)
        rcnn_delta_weights = tf.concat(rcnn_delta_weights, axis=0)
        
        return rcnn_rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights
    
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape, batch_ind):
        '''
        Args
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.
            
        Returns
        ---
            rois: [num_rois, (batch_ind, y1, x1, y2, x2)]
            target_matchs: [num_rois]
            target_deltas: [num_rois, (dy, dx, log(dh), log(dw))]
        '''
        H, W = img_shape
        
        
        trimmed_proposals, _ = trim_zeros(proposals[:, 1:])
        
        gt_boxes, non_zeros = trim_zeros(gt_boxes)
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)
        
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        
        overlaps = geometry.compute_overlaps(trimmed_proposals, gt_boxes)
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        
        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]
        
        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.num_rcnn_deltas * self.positive_fraction)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.positive_fraction
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        labels = tf.gather(gt_class_ids, roi_gt_box_assignment)
        
        
        delta_targets = transforms.bbox2delta(positive_rois[:, 1:], roi_gt_boxes, self.target_means, self.target_stds)
        
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        
        
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.num_rcnn_deltas - tf.shape(rois)[0], 0)
        num_bfg = rois.shape[0]
        
        rois = tf.pad(rois, [(0, P), (0, 0)])
        labels = tf.pad(labels, [(0, N)], constant_values=0)
        labels = tf.pad(labels, [(0, P)], constant_values=0)
        delta_targets = tf.pad(delta_targets, [(0, N + P), (0, 0)])
        
        
        # Compute weights
        if num_bfg > 0:
            label_weights = tf.concat([tf.ones((num_bfg,), dtype=tf.float32) / num_bfg, 
                                       tf.zeros((P,), dtype=tf.float32)], 
                                      axis=0)
            delta_weights = tf.concat([tf.ones((num_bfg - N,), dtype=tf.float32) / num_bfg, 
                                       tf.zeros((N + P,), dtype=tf.float32)], 
                                      axis=0)
        else:
            label_weights = tf.zeros((labels.shape[0],), dtype=tf.float32)
            delta_weights = tf.zeros((delta_targets.shape[0],), dtype=tf.float32)

        delta_weights = tf.tile(tf.reshape(delta_weights, (-1, 1)), [1, 4])
        
        new_delta_targets = tf.zeros((labels.shape[0], self.num_classes, 4))
        new_delta_weights = tf.zeros((labels.shape[0], self.num_classes, 4))

        ids = tf.stack([tf.range(self.num_rcnn_deltas, dtype=tf.int64), labels], axis=1)
        delta_targets = tf.tensor_scatter_nd_update(new_delta_targets,
                                                    ids,
                                                    delta_targets)
        
        delta_weights = tf.tensor_scatter_nd_update(new_delta_weights,
                                                    ids,
                                                    delta_weights)
        
        return rois, labels, label_weights, delta_targets, delta_weights