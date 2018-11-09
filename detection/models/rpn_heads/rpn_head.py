import tensorflow as tf
layers = tf.keras.layers

from detection.core.bbox import transforms
from detection.utils.misc import *

class RPNHead(tf.keras.Model):
    def __init__(self, 
                 anchors_per_location, 
                 proposal_count=2000, 
                 nms_threshold=0.7, 
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2), **kwags):
        '''Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchors_per_location: int. the number of anchors per pixel 
                in the feature maps.
            proposal_count: int. RPN proposals kept after non-maximum 
                supression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
        '''
        super(RPNHead, self).__init__(**kwags)
        
        self.anchors_per_location = anchors_per_location
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = tf.constant(target_means)
        self.target_stds = tf.constant(target_stds)
        
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal', 
                                             name='rpn_conv_shared')
        
        self.rpn_class_raw = layers.Conv2D(2 * anchors_per_location, (1, 1),
                                           kernel_initializer='he_normal', 
                                           name='rpn_class_raw')

        self.rpn_delta_pred = layers.Conv2D(anchors_per_location * 4, (1, 1),
                                           kernel_initializer='he_normal', 
                                            name='rpn_bbox_pred')
        
    def __call__(self, inputs, training=True):
        '''
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                one level of pyramid feat-maps.
        
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        
        shared = self.rpn_conv_shared(inputs)
        shared = tf.nn.relu(shared)
        
        x = self.rpn_class_raw(shared)
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits)
        
        x = self.rpn_delta_pred(shared)
        rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
        
        return [rpn_class_logits, rpn_probs, rpn_deltas]
    
    def get_proposals(self, rpn_probs, rpn_deltas, anchors, valid_flags, img_metas):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in pixel 
                coordinates.
            valid_flags: [batch_size, num_anchors]
            img_metas: [batch_size, 11]
        
        Returns
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in 
                normalized coordinates.
        
        Note that num_proposals is no more than proposal_count. And different 
           images in one batch may have different num_proposals.
        '''
        
        rpn_probs = rpn_probs[:, :, 1]
        
        img_shapes = calc_img_shapes(img_metas)
        
        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], img_shapes[i])
            for i in range(img_metas.shape[0])
        ]
        
        return proposals_list  
        
    
    def _get_proposals_single(self, rpn_probs, rpn_deltas, anchors, valid_flags, img_shape):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
                pixel coordinates.
            valid_flags: [num_anchors]
            img_shape: np.ndarray. [2]. (img_height, img_width)
        
        Returns
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in normalized 
                coordinates.
        '''
        
        H, W = img_shape
        
        # filter invalid anchors
        valid_flags = tf.cast(valid_flags, tf.bool)
        
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)

        # Improve performance
        pre_nms_limit = min(6000, anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)
        
        # Get refined anchors
        proposals = transforms.delta2bbox(anchors, rpn_deltas, 
                                          self.target_means, self.target_stds)
        
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = transforms.bbox_clip(proposals, window)
        
        # Normalize
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # NMS
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, indices)
        
        return proposals
        
    def compute_output_shape(self, input_shape):
        batch, height, width, channel = input_shape.as_list()
        
        return [tf.TensorShape([batch, height * width * self.anchors_per_location, 2]),
                tf.TensorShape([batch, height * width * self.anchors_per_location, 2]),
                tf.TensorShape([batch, height * width * self.anchors_per_location, 4])]
        