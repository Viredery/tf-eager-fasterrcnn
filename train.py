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

train_dataset = coco.CocoDataSet('./COCO2017/', 'train',
                                 num_max_gts=1000,
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=(123.675, 116.28, 103.53),
                                 std=(58.395, 57.12, 57.375),
                                 scale=(800, 1024))

train_generator = data_generator.DataGenerator(train_dataset)


batch_size = 2

train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.padded_batch(
    batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))

# model
model = faster_rcnn.FasterRCNN(
    num_classes=len(train_dataset.get_categories()))



# train
optimizer = tf.train.AdamOptimizer(1e-4)

iterator = train_tf_dataset.make_one_shot_iterator()
loss_history = []
for (batch, inputs) in enumerate(iterator):
    
    imgs, img_metas, bboxes, labels = inputs
    with tf.GradientTape() as tape:
        rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
            model((imgs, img_metas, bboxes, labels), training=True)
        
        loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    print(batch, '-', imgs.shape, loss_value.numpy())
    loss_history.append(loss_value.numpy())
