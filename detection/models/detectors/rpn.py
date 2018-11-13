import tensorflow as tf

from detection.models.backbones import resnet
from detection.models.necks import fpn
from detection.models.rpn_heads import rpn_head

from detection.core.anchor import anchor_generator, anchor_target
from detection.core.loss import losses

class RPN(tf.keras.Model):
    def __init__(self, **kwags):
        super(RPN, self).__init__(**kwags)
        
        # Anchor attributes
        self.ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)
        
        # The strides of each layer of the FPN Pyramid.
        self.FEATURE_STRIDES = (4, 8, 16, 32, 64)
        
        # Bounding box refinement mean and standard deviation
        self.RPN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
        
        self.backbone = resnet.ResNet(depth=101)
        self.neck = fpn.FPN()
        self.rpn_head = rpn_head.RPNHead(anchors_per_location=len(self.ANCHOR_RATIOS))
        
        self.generator = anchor_generator.AnchorGenerator(
            scales=self.ANCHOR_SCALES, 
            ratios=self.ANCHOR_RATIOS, 
            feature_strides=self.FEATURE_STRIDES)
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=self.RPN_TARGET_MEANS, 
            target_stds=self.RPN_TARGET_STDS)
        
        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss
    
    
    def __call__(self, inputs, training=True):

        if training: # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs
            
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)
        
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn_head(p, training=training))
        
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1)
                   for o in outputs]
        
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        
        if training:
            anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

            rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
                anchors, valid_flags, gt_boxes, gt_class_ids)

            rpn_class_loss = self.rpn_class_loss(
                rpn_target_matchs, rpn_class_logits)

            rpn_bbox_loss = self.rpn_bbox_loss(
                rpn_target_deltas, rpn_target_matchs, rpn_deltas)

            return [rpn_class_loss, rpn_bbox_loss]
        else:
            return [rpn_class_logits, rpn_probs, rpn_deltas]
        
        