import tensorflow as tf
from nets import inception, resnet_v1, resnet_v2, resnet_utils, vgg, mobilenet_v1
from preprocessing.vgg_preprocessing import preprocess_image as vgg_preprocess
from preprocessing.inception_preprocessing import preprocess_image as inception_preprocess
import numpy as np

slim = tf.contrib.slim

import nasnet
import nasnet_utils

class NASNetLargeModel(object):
  """
  NASNet Large model.
  """

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits =  None
    self.ckpt = 'model.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """ Construct the model and return probablities for given input ."""
    if self.built:
      return self.logits
    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = nasnet.build_nasnet_large(
          x_input, self.num_classes, is_training=False)
    self.built = True
    self.scores = end_points['Predictions']
    self.logits = logits
    self.preds = tf.argmax(logits, axis=1)
    return self.logits
    
class InceptionV3Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'inception_v3.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = inception.inception_v3(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)
        
        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.scores = end_points['predictions']
    self.logits = logits
    self.preds = preds
    #output = end_points['logits']
    # Strip off the extra reshape op at the output
    #probs = output.op.inputs[0]
    return logits

class AdvInceptionV3Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'adv_inception_v3.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = inception.inception_v3(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.scores = end_points['predictions']
    self.logits = logits
    self.preds = preds
    #output = end_points['logits']
    # Strip off the extra reshape op at the output
    #probs = output.op.inputs[0]
    return logits


class InceptionV4Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'inception_v4.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v4_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = inception.inception_v4(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.scores = end_points['Predictions']
    self.logits = logits
    self.preds = preds
    return logits


class InceptionResNetV2Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes, batch_size=None):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'inception_resnet_v2_2016_08_30.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = inception.inception_resnet_v2(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

      preds = tf.argmax(logits, axis=1)
    self.built = True
    print(end_points.keys())
    self.scores = end_points['Predictions']
    self.logits = logits
    self.preds = preds
    return logits

class EnsAdvInceptionResNetV2Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes, batch_size=None):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = inception.inception_resnet_v2(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.scores = end_points['Predictions']
    self.logits = logits
    self.preds = preds
    return logits

class ResNetV1Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'resnet_v1_50.ckpt'

  def __call__(self, x_input, batch_size, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    # ResNet V1 and VGG have different preprocessing
    preproc = tf.map_fn(
      lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                 resnet_v1.resnet_v1.default_image_size,
                                 resnet_v1.resnet_v1.default_image_size), x_input)

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = resnet_v1.resnet_v1_50(
            preproc, num_classes=self.num_classes - 1, is_training=is_training,
            reuse=reuse)

        # VGG and ResNetV1 don't have a background class
        background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
        logits = tf.concat([background_class, logits], axis=1)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.scores = end_points['Predictions']
    self.logits = logits
    self.preds = preds
    return logits


class ResNetV1_152_Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'resnet_v1_152.ckpt'

  def __call__(self, x_input, batch_size, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    # ResNet V1 and VGG have different preprocessing
    preproc = tf.map_fn(
      lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                 resnet_v1.resnet_v1.default_image_size,
                                 resnet_v1.resnet_v1.default_image_size), x_input)

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = resnet_v1.resnet_v1_152(
            preproc, num_classes=self.num_classes - 1, is_training=is_training,
            reuse=reuse)

        # VGG and ResNetV1 don't have a background class
        background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
        logits = tf.concat([background_class, logits], axis=1)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.logits = logits
    self.scores = tf.concat([tf.constant(0, dtype=tf.float32, shape=[batch_size,1]), end_points['predictions']], axis=1)
    self.preds = preds
    return logits


class ResNetV2Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'resnet_v2_50.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):

    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = resnet_v2.resnet_v2_50(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.logits = logits
    self.scores = end_points['Predictions']
    self.preds = preds
    return logits

class ResNetV2_101_Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'resnet_v2_101.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):

    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = resnet_v2.resnet_v2_101(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    self.logits = logits
    self.scores = end_points['Predictions']
    self.preds = preds
    return logits


class ResNetV2_152_Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'resnet_v2_152.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):

    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with tf.variable_scope(self.ckpt):
        logits, end_points = resnet_v2.resnet_v2_152(
            x_input, num_classes=self.num_classes, is_training=is_training,
            reuse=reuse)

        preds = tf.argmax(logits, axis=1)
    self.built = True
    print(end_points.keys())
    self.scores = end_points['predictions']
    self.logits = logits
    self.preds = preds
    return logits


class VGG16(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'vgg_16.ckpt'

  def __call__(self, x_input, batch_size, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    # ResNet V1 and VGG have different preprocessing
    preproc = tf.map_fn(
      lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                 vgg.vgg_16.default_image_size,
                                 vgg.vgg_16.default_image_size), x_input)
    with tf.variable_scope(self.ckpt):
      logits, end_points = vgg.vgg_16(
          preproc, num_classes=self.num_classes - 1, is_training=is_training)

      # VGG and ResNetV1 don't have a background class
      background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
      logits = tf.concat([background_class, logits], axis=1)

      preds = tf.argmax(logits, axis=1)

    self.built = True
    self.logits = logits
    self.preds = preds
    return logits


class MobileNetModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
    self.logits = None
    self.ckpt = 'mobilenet_v1_1.0_224.ckpt'

  def __call__(self, x_input, batch_size=None, is_training=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    preproc = tf.map_fn(
      lambda img: inception_preprocess(img,
                                       mobilenet_v1.mobilenet_v1.default_image_size,
                                       mobilenet_v1.mobilenet_v1.default_image_size), x_input)

    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          preproc, num_classes=self.num_classes, is_training=is_training,
          reuse=reuse)

      preds = tf.argmax(logits, axis=1)
    self.built = True
    self.logits = logits
    self.preds = preds
    return logits
