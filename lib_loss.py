
import numpy as np
import tensorflow as tf

def huber_loss(labels, predictions, delta):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    loss = tf.where(condition, small_res, large_res)
    loss = tf.reduce_mean(loss)
    return loss

def logcosh(true, pred):
    loss = tf.log(tf.cosh(pred - true))
    return tf.reduce_mean(loss)

def quan(true, pred, theta):
    loss = tf.where(true >= pred, theta*(tf.abs(true-pred)), (1-theta)*(tf.abs(true-pred)))
    return tf.reduce_mean(loss)


def TotalVariation(images, name = None):
    '''
    Calculate and return the total variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the
    images.

    This can be used as a loss-function during optimization so as to suppress
    noise in images. If you have a batch of images, then you should calculate
    the scalar loss-value as the sum:
    `loss = tf.reduce_sum(tf.image.total_variation(images))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or
                3-D Tensor of shape `[height, width, channels]`.

        name: A name for the operation (optional).

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.

        If `images` was 4-D, return a 1-D float Tensor of shape `[batch]`
        with the total variation for each image in the batch.
        If `images` was 3-D, return a scalar float with the total variation for
        that image.
    '''

    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 3:

            pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
            pixel_dif1 = array_ops.pad(
                pixel_dif1, paddings = [[0,1], [0,0], [0,0]]
            )
            pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
            pixel_dif2 = array_ops.pad(
                pixel_dif2, paddings = [[0,0], [0,1], [0,0]]
            )

        elif ndims == 4:

            pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
            pixel_dif1 = array_ops.pad(
                pixel_dif1, paddings = [[0,0], [0,1], [0,0], [0,0]]
            )
            pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
            pixel_dif2 = array_ops.pad(
                pixel_dif2, paddings = [[0,0], [0,0], [0,1], [0,0]]
            )

        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences
        tot_var = math_ops.abs(pixel_dif1) + math_ops.abs(pixel_dif2)
        return tot_var
# def TotalVariation(inputs):
#     """
#     A smooth loss in fact. Like the smooth prior in MRF.
#     V(y) = || y_{n+1} - y_n ||_2
#     :param inputs:
#     :return:
#     """
#     # dy = inputs[:, :-1, ...] - inputs[:, 1:, ...]
#     # dx = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
#     dy = inputs[:-1, :] - inputs[1:, :]
#     dx = inputs[:, :-1] - inputs[:, 1:]
#     size_dy = tf.size(dy, out_type=tf.float32)
#     size_dx = tf.size(dx, out_type=tf.float32)
#     return tf.nn.l2_loss(dy) / size_dy + tf.nn.l2_loss(dx) / size_dx
def losses(logits, labels, name = 'losses'):
    with tf.name_scope(name):

        change1 = labels <=20
        change1 = tf.cast(change1,tf.float32)
        Y1 = change1*0.05

        change2 = labels > 20
        change2 = tf.cast(change2,tf.float32)
        # Y2 = tf.multiply(labels, change2)
        Y2 = change2*1
        Y_change = Y1+Y2

        loss = (tf.reduce_mean(tf.square(logits - labels)/Y_change)+ 0.01* tf.reduce_mean(TotalVariation(logits)))
        # loss = (tf.reduce_mean(tf.square(logits - labels)) + 0.1 * tf.reduce_mean(TotalVariation(logits)))
            # FLAGS.BETA *  tf.reduce_mean(TotalVariation(logits) * mask)
        return loss