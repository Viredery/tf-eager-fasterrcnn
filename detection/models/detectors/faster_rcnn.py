import tensorflow as tf

from detection.models.backbones import resnet
from detection.models.necks import fpn
from detection.models.rpn_heads import rpn_head

from detection.models.bbox_heads import bbox_head
from detection.models.roi_extractors import roi_align


from detection.core.anchor import anchor_generator, anchor_target
from detection.core.loss import losses

from detection.core.bbox import bbox_target, transforms

class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwags):
        super(FasterRCNN, self).__init__(**kwags)
       
        self.NUM_CLASSES = num_classes
        
        # Anchor attributes
        self.ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)
        
        # The strides of each layer of the FPN Pyramid.
        self.FEATURE_STRIDES = (4, 8, 16, 32, 64)
        
        # Bounding box refinement mean and standard deviation
        self.RPN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
        
        self.PRN_PROPOSAL_COUNT = 2000
        self.PRN_NMS_THRESHOLD = 0.7
        
        self.ROI_BATCH_SIZE = 512
        
        # Bounding box refinement mean and standard deviation
        self.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
        
        self.POOL_SIZE = (7, 7)
        
        
        self.backbone = resnet.ResNet50()
        self.neck = fpn.FPN()
        self.rpn_head = rpn_head.RPNHead(anchors_per_location=len(self.ANCHOR_RATIOS),
                                         proposal_count=self.PRN_PROPOSAL_COUNT,
                                         nms_threshold=self.PRN_NMS_THRESHOLD,
                                         target_means=self.RPN_TARGET_MEANS,
                                         target_stds=self.RPN_TARGET_STDS)
        
        self.roi_align = roi_align.PyramidROIAlign(pool_shape=self.POOL_SIZE)
        self.bbox_head = bbox_head.BBoxHead(num_classes=self.NUM_CLASSES,
                                            pool_size=self.POOL_SIZE)
        
        self.generator = anchor_generator.AnchorGenerator(
            scales=self.ANCHOR_SCALES, 
            ratios=self.ANCHOR_RATIOS, 
            feature_strides=self.FEATURE_STRIDES)
        
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=self.RPN_TARGET_MEANS, 
            target_stds=self.RPN_TARGET_STDS)
        
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS, 
            num_rcnn_deltas=self.ROI_BATCH_SIZE)
        
        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss
        
        self.rcnn_class_loss = losses.rcnn_class_loss
        self.rcnn_bbox_loss = losses.rcnn_bbox_loss

    def __call__(self, inputs, training=True):      
        if training: # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs

        C2, C3, C4, C5 = self.backbone(imgs, 
                                       training=training)
        
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], 
                                       training=training)
        
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn_head(p, training=training))
        
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, anchors, valid_flags, img_metas)
        
        if training:
            rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list = \
                self.bbox_target.build_targets(
                    proposals_list, gt_boxes, gt_class_ids, img_metas)
        else:
            rois_list = proposals_list
            
        pooled_regions_list = self.roi_align(
            (rois_list, rcnn_feature_maps, img_metas), training=training)

        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = \
            self.bbox_head(pooled_regions_list, training=training)

        if training:
            rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
                anchors, valid_flags, gt_boxes, gt_class_ids)
            
            rpn_class_loss = self.rpn_class_loss(
                rpn_target_matchs, rpn_class_logits)
            rpn_bbox_loss = self.rpn_bbox_loss(
                rpn_target_deltas, rpn_target_matchs, rpn_deltas)

            rcnn_class_loss = self.rcnn_class_loss(
                rcnn_target_matchs_list, rcnn_class_logits_list)
            rcnn_bbox_loss = self.rcnn_bbox_loss(
                rcnn_target_deltas_list, rcnn_target_matchs_list, rcnn_deltas_list)            
            
            
            return [rpn_class_loss, rpn_bbox_loss, 
                    rcnn_class_loss, rcnn_bbox_loss]
        else:
            detections_list = self.bbox_head.get_bboxes(
                rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)
        
            return self.unmold_detections(detections_list, img_metas)

    def unmold_detections(self, detections_list, img_metas):
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
