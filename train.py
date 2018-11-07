import os
import tensorflow as tf

# eager execution
tf.enable_eager_execution()
tf.executing_eagerly()

# tensorflow config - using one gpu and extending the GPU 
# memory region needed by the TensorFlow process
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn

batch_size = 2

# train dataset 
train_dataset = coco.CocoDataSet('./COCO2017/', 'train',
                                 num_max_gts=1000,
                                 flip_ratio=0.5,
                                 size_divisor=64,
                                 mean=(123.675, 116.28, 103.53),
                                 std=(58.395, 57.12, 57.375),
                                 scale=(800, 1024))


generator = data_generator.DataGenerator(train_dataset)

train_dataset = tf.data.Dataset.from_generator(generator, 
                                    (tf.float32, tf.float32, tf.float32, tf.int32))

train_dataset = train_dataset.padded_batch(batch_size, 
                                           padded_shapes=([None,None,None],[None],[None,None],[None]))

# model
model = faster_rcnn.FasterRCNN(num_classes=81)



# train
optimizer = tf.train.AdamOptimizer(1e-4)

iterator = train_dataset.make_one_shot_iterator()

loss_history = []
for (batch, inputs) in enumerate(iterator):
    
    imgs, img_metas, bboxes, labels = inputs
    with tf.GradientTape() as tape:
        outputs = model((imgs, img_metas, bboxes, labels), training=True)
        rpn_class_logits, rpn_probs, rpn_deltas, \
            rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list, \
            rcnn_target_matchs_list, rcnn_target_deltas_list = outputs
        

        
        loss_value = model.loss(img_metas, bboxes, labels, rpn_class_logits, rpn_probs, rpn_deltas,
                                rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list,
                                rcnn_target_matchs_list, rcnn_target_deltas_list)

    print(batch, '-', imgs.shape, loss_value.numpy())
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())