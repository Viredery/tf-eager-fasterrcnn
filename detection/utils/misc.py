import tensorflow as tf

def trim_zeros(boxes, name=None):
    '''Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def parse_image_meta(meta):
    '''Parses a tensor that contains image attributes to its components.
    
    Args
    ---
        meta: [batch, 11]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    ori_shape = meta[:, 0:3]
    img_shape = meta[:, 3:6]
    pad_shape = meta[:, 6:9]
    scale = meta[:, 9]  
    flip = meta[:, 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }

def batch_slice(inputs, graph_fn, batch_size, names=None):
    '''Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    '''
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result