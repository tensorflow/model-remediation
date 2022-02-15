# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of MMD Loss."""

from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.min_diff.losses import base_loss


@tf.keras.utils.register_keras_serializable()
class MMDLoss(base_loss.MinDiffLoss):

  # pyformat: disable
  """Maximum Mean Discrepancy between predictions on two groups of examples.

  Arguments:
    kernel: String (name of kernel) or `losses.MinDiffKernel` instance to be
      applied on the predictions. Defaults to `'gaussian'` and it is recommended
      that this be either
      `'gaussian'`
      (`min_diff.losses.GaussianKernel`) or `'laplacian'`
      (`min_diff.losses.LaplacianKernel`).
    predictions_transform: Optional transform function to be applied to the
      predictions. This can be used to smooth out the distributions or limit the
      range of predictions.

      The choice of whether to apply a transform to the predictions is task and
      data dependent. For example, for classifiers, it might make sense to apply
      a `tf.sigmoid` transform to the predictions (if this is not done already)
      so that MMD is calculated in probability space rather than on raw
      predictions. In some cases, such as regression, not having any transform
      is more likely to yield successful results.
    name: Name used for logging and tracking. Defaults to `'mmd_loss'`.
    enable_summary_histogram: Optional bool indicating if `tf.summary.histogram`
      should be included within the loss. Defaults to True.

  The Maximum Mean Discrepancy (MMD) is a measure of the distance between the
  distributions of prediction scores on two groups of examples. The metric
  guarantees that the result is 0 if and only if the two distributions it is
  comparing are exactly the same.

  The `membership` input indicates with a numerical value whether
  each example is part of the sensitive group with a numerical value. This
  currently only supports hard membership of `0.0` or `1.0`.

  For more details, see the
  [paper](http://papers.nips.cc/paper/3110-a-kernel-method-for-the-two-sample-problem.pdf).
  """
  # pyformat: enable

  def __init__(self,
               kernel="gaussian",
               predictions_transform=None,
               name: Optional[str] = None,
               enable_summary_histogram: Optional[bool] = True):
    """Initialize an instance of MMDLoss."""
    super(MMDLoss, self).__init__(
        predictions_transform=predictions_transform,
        predictions_kernel=kernel,
        name=name or "mmd_loss",
        enable_summary_histogram=enable_summary_histogram)

  def _preprocess(self,
                  membership: types.TensorType,
                  predictions: types.TensorType,
                  sample_weight: Optional[types.TensorType] = None):
    """Preprocess inputs before computing losses."""
    membership, predictions, normed_weights = self._preprocess_inputs(
        membership, predictions, sample_weight)
    _, predictions_kernel = self._apply_kernels(membership, predictions)

    pos_mask = tf.cast(tf.equal(membership, 1.0), tf.float32)
    neg_mask = tf.cast(tf.equal(membership, 0.0), tf.float32)

    return predictions_kernel, normed_weights, pos_mask, neg_mask

  def _calculate_mean(self, predictions_kernel: types.TensorType,
                      normed_weights: types.TensorType,
                      pos_mask: types.TensorType, neg_mask):
    """Calculate means of groups."""
    weights_ij = tf.matmul(normed_weights, tf.transpose(normed_weights))

    pos_mean_mask = tf.matmul(pos_mask, tf.transpose(pos_mask))
    pos_mean_weights = weights_ij * pos_mean_mask
    neg_mean_mask = tf.matmul(neg_mask, tf.transpose(neg_mask))
    neg_mean_weights = weights_ij * neg_mean_mask
    pos_neg_mean_mask = tf.matmul(pos_mask, tf.transpose(neg_mask))
    pos_neg_mean_weights = weights_ij * pos_neg_mean_mask

    pos_mean = tf.math.divide_no_nan(
        tf.reduce_sum(pos_mean_weights * predictions_kernel),
        tf.reduce_sum(pos_mean_weights))
    neg_mean = tf.math.divide_no_nan(
        tf.reduce_sum(neg_mean_weights * predictions_kernel),
        tf.reduce_sum(neg_mean_weights))
    pos_neg_mean = tf.math.divide_no_nan(
        tf.reduce_sum(pos_neg_mean_weights * predictions_kernel),
        tf.reduce_sum(pos_neg_mean_weights))

    return pos_mean, neg_mean, pos_neg_mean

  def call(self,
           membership: types.TensorType,
           predictions: types.TensorType,
           sample_weight: Optional[types.TensorType] = None):
    """Computes MMD the loss value."""
    predictions_kernel, normed_weights, pos_mask, neg_mask = self._preprocess(
        membership, predictions, sample_weight)

    pos_mean, neg_mean, pos_neg_mean = self._calculate_mean(
        predictions_kernel, normed_weights, pos_mask, neg_mask)

    # MMD is actually the square root of the following quatity. However, the
    # derivative of sqrt is easy to blow up when the value is close to 0. So we
    # do not use that.
    loss = pos_mean - 2 * pos_neg_mean + neg_mean
    return loss

  @classmethod
  def from_config(cls, config):
    """Creates a MMDLoss instance fron the config."""
    config = cls._deserialize_config(config)
    # Rename 'predictions_kernel' to 'kernel'
    if "predictions_kernel" in config:
      config["kernel"] = config["predictions_kernel"]
      del config["predictions_kernel"]
    return cls(**config)
