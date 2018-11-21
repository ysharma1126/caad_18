"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pdb
import numpy as np
from scipy.misc import imread
from scipy.stats import mode
import random
import tensorflow as tf

from PIL import Image
import StringIO
import cv2 as cv

slim = tf.contrib.slim

import time
tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
        'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
        'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
        'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
        'batch_size', 20, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    kernel = np.ones((3,3),np.float32)/9
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0


        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)
    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, 
        tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], 
            [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)



def main(_):
    start_time = time.time()    
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        img_resize_tensor = tf.placeholder(tf.int32, [2])
        x_input_resize = tf.image.resize_images(x_input, img_resize_tensor, 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        shape_tensor = tf.placeholder(tf.int32, [3])
        padded_input = padding_layer_iyswim(x_input_resize, shape_tensor)
        # 330 is the last value to keep 8*8 output, 362 is the last value to keep 9*9 output, stride = 32
        padded_input.set_shape(
            (FLAGS.batch_size, FLAGS.image_resize, FLAGS.image_resize, 3))
        import models

        initialized_vars = set()
        savers = []

        # list of models in our ensemble
        all_models = [
            models.NASNetLargeModel,
            models.EnsAdvInceptionResNetV2Model,
        ]

        # build all the models and specify the saver
        for i, model in enumerate(all_models):
            all_models[i] = model(num_classes)
            all_models[i](padded_input, FLAGS.batch_size)
            all_vars = slim.get_model_variables()
            model_vars = [k for k in all_vars if k.name.startswith(all_models[i].ckpt)]
            var_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in model_vars}
            savers.append(tf.train.Saver(var_dict))

        itr = 18
        kernel_size = 10
        p_dropout = 0.1
        avoid = True
        with tf.Session() as sess:
            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for model, saver in zip(all_models, savers):
                    saver.restore(sess, FLAGS.checkpoint_path + '/' + model.ckpt)

                pred = [model.scores for model in all_models]
                kernel = np.ones((kernel_size,kernel_size),np.float32)/np.square(kernel_size)
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    scores_list = []
                    for _ in range(itr):
                        images_ = images.copy()
                        for i, img in enumerate(images_):
                            rand = np.random.uniform(size=img.shape) < p_dropout
                            image_d = img * rand
                            image_s = cv.filter2D(img,-1,kernel)
                            image = (image_d) + (image_s * (1-rand))
                            images_[i] = image
                        resize_shape_ = np.random.randint(310, 331)
                        preds = sess.run(pred, 
                            feed_dict={x_input: images_, 
                                       img_resize_tensor: [resize_shape_]*2,
                                       shape_tensor: np.array([random.randint(0, 
                                                                FLAGS.image_resize - resize_shape_),
                                                               random.randint(0, 
                                                                FLAGS.image_resize - resize_shape_),
                                                               FLAGS.image_resize])})
                        preds = np.array(preds)
                        scores_list.append(preds)
                    scores_list = np.array(scores_list)
                    scores_models = np.mean(scores_list, 0)# mean across all iterations
                    scores_final = np.mean(scores_models, 0) # mean across all models
                    labels = np.argmax(scores_final, axis=1)
                    if avoid:
                        fool_label = 612  # index 0 611: 'jigsaw puzzle'

                        labels = list(labels)
                        for i,label in enumerate(labels):
                            if label == fool_label:
                                ordered_labels = scores_final[i].argsort()[-2:][::-1]
                                labels[i] = ordered_labels[-1] # The second label
                        labels = np.array(labels)
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))
        end_time = time.time()
        print('took ', (end_time - start_time))
if __name__ == '__main__':
    tf.app.run()
