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
                 neg_iou_thr=0.5):
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
        
        rois_list = []
        rcnn_target_matchs_list = []
        rcnn_target_deltas_list = []
        
        for i in range(batch_size):
            rois, target_matchs, target_deltas = self._build_single_target(
                proposals[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i], i)
            rois_list.append(rois)
            rcnn_target_matchs_list.append(target_matchs)
            rcnn_target_deltas_list.append(target_deltas)


        rois = tf.concat(rois_list, axis=0)
        target_matchs = tf.concat(rcnn_target_matchs_list, axis=0)
        target_deltas = tf.concat(rcnn_target_deltas_list, axis=0)
        
        rois = tf.stop_gradient(rois)
        target_matchs = tf.stop_gradient(target_matchs)
        target_deltas = tf.stop_gradient(target_deltas)
        
        return rois, target_matchs, target_deltas
    
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
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.positive_fraction
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        target_matchs = tf.gather(gt_class_ids, roi_gt_box_assignment)
        
        
        target_deltas = transforms.bbox2delta(positive_rois[:, 1:], roi_gt_boxes, self.target_means, self.target_stds)
        
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        
        
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.num_rcnn_deltas - tf.shape(rois)[0], 0)
        
        rois = tf.pad(rois, [(0, P), (0, 0)])
        
        target_matchs = tf.pad(target_matchs, [(0, N)])
        target_matchs = tf.pad(target_matchs, [(0, P)], constant_values=-1)
        target_deltas = tf.pad(target_deltas, [(0, N + P), (0, 0)])

        return rois, target_matchs, target_deltas