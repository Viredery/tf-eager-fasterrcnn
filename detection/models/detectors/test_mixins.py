import numpy as np
import tensorflow as tf

from detection.core.bbox import transforms
from detection.utils.misc import *

class RPNTestMixin(object):
    
    def _unmold_proposals(self, proposals_list, img_metas):
        return [
            self._unmold_single_proposal(proposals_list[i], img_metas[i])
            for i in range(img_metas.shape[0])
        ]

    def _unmold_single_proposal(self, proposals, img_meta):
        boxes = proposals[:, :4]
        scores = proposals[:, 4]

        H, W = calc_pad_shapes(img_meta)
        boxes = boxes * tf.constant([H, W, H, W], dtype=tf.float32)
        
        boxes = transforms.bbox_mapping_back(boxes, img_meta)

        if self.RCNN_MAX_INSTANCES is not None:
            ix = tf.nn.top_k(scores, self.RCNN_MAX_INSTANCES).indices[::-1]

            boxes = tf.gather(boxes, ix)
            scores = tf.gather(scores, ix)

        return {'proposals': boxes.numpy(),
                'scores': scores.numpy()}    
    
    def simple_test_rpn(self, img, img_meta):
        '''
        Args
        ---
            imgs: np.ndarray. [height, width, channel]
            img_metas: np.ndarray. [11]
        
        '''
        imgs = tf.Variable(np.expand_dims(img, 0))
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))

        x = self.backbone(imgs, training=False)
        x = self.neck(x, training=False)
        
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(x, training=False)
        
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, with_probs=True)

        return self._unmold_proposals(proposals_list, img_metas)[0]
    
class BBoxTestMixin(object):
    
    def _unmold_detections(self, detections_list, img_metas):
        return [
            self._unmold_single_detection(detections_list[i], img_metas[i])
            for i in range(img_metas.shape[0])
        ]


    def _unmold_single_detection(self, detections, img_meta):
        zero_ix = tf.where(tf.not_equal(detections[:, 4], 0))
        detections = tf.gather_nd(detections, zero_ix)

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:, :4]
        class_ids = tf.cast(detections[:, 4], tf.int32)
        scores = detections[:, 5]

        boxes = transforms.bbox_mapping_back(boxes, img_meta)

        return {'rois': boxes.numpy(),
                'class_ids': class_ids.numpy(),
                'scores': scores.numpy()}

    def simple_test_bboxes(self, img, img_meta):
        '''
        Args
        ---
            imgs: np.ndarray. [height, width, channel]
            img_metas: np.ndarray. [11]
        
        '''
        imgs = tf.Variable(np.expand_dims(img, 0))
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))
        
        x = self.backbone(imgs, training=False)
        P2, P3, P4, P5, P6 = self.neck(x, training=False)
        
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=False)
        
        rois_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, with_probs=False)

        pooled_regions_list = self.roi_align(
            (rois_list, rcnn_feature_maps, img_metas), training=False)

        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = \
            self.bbox_head(pooled_regions_list, training=False)
        
        detections_list = self.bbox_head.get_bboxes(
            rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)
        
        return self._unmold_detections(detections_list, img_metas)[0]