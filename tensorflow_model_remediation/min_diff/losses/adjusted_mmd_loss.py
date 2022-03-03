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

"""Implementation of Adjusted MMD Loss."""

from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.min_diff.losses.mmd_loss import MMDLoss

# Epsilon constant used to represent small quantity.
_EPSILON = 1e-2


class AdjustedMMDLoss(MMDLoss):

  # pyformat: disable
  """Adjusted Maximum Mean Discrepancy between predictions on two groups of examples.

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
    name: Name used for logging and tracking. Defaults to `'adjusted_mmd_loss'`.
    enable_summary_histogram: Optional bool indicating if `tf.summary.histogram`
      should be included within the loss. Defaults to True.

  The main motivation for adjusted MMDLoss is to capture variances of each
  membership's predictions. In the adjusted MMDLoss, we calculate the sum of
  variances of mean for each membership's prediction, and divide the original
  MMDLoss with the sum of variances. The adjustment works for any kernel.
  """
  # pyformat: enable

  def __init__(self,
               kernel="gaussian",
               predictions_transform=None,
               name: Optional[str] = None,
               enable_summary_histogram: Optional[bool] = True):
    """Initialize an instance of AdjustedMMDLoss."""
    super(AdjustedMMDLoss, self).__init__(
        kernel=kernel,
        predictions_transform=predictions_transform,
        name=name or "adjusted_mmd_loss",
        enable_summary_histogram=enable_summary_histogram)

  def _calculate_var_of_mean(self, predictions_kernel: types.TensorType,
                             normed_weights: types.TensorType,
                             pos_mask: types.TensorType,
                             neg_mask: types.TensorType, pos_mean, neg_mean):
    """Calculate variance of mean of each group."""
    # We want to adjust the MMD loss by taking into account the variance of the
    # predictions. In particular, in the product kernel case where k(x, y) = xy,
    # MMD_loss^2 = (mu_0 - mu_1)^2. However, how big the MMD loss is is relative
    # to the variance of our estimate of mu_0 and mu_1.
    # Therefore, we calculate the variance of mean of each group and use the
    # sum as the denominator to adjust MMD loss.
    # See standard error of the mean:
    # https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean
    # When we have weighted examples, see weighted mean:
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Simple_i.i.d_case

    # Calculate square mean of each group.
    pos_diag_weights = tf.linalg.diag(
        tf.reshape(normed_weights * pos_mask, [-1]))
    neg_diag_weights = tf.linalg.diag(
        tf.reshape(normed_weights * neg_mask, [-1]))

    pos_square_mean = tf.math.divide_no_nan(
        tf.reduce_sum(pos_diag_weights * predictions_kernel),
        tf.reduce_sum(pos_diag_weights))

    neg_square_mean = tf.math.divide_no_nan(
        tf.reduce_sum(neg_diag_weights * predictions_kernel),
        tf.reduce_sum(neg_diag_weights))

    # Calculate variance of each group.
    pos_var = pos_square_mean - pos_mean
    neg_var = neg_square_mean - neg_mean

    # Calculate variance of mean of each group.
    pos_n = tf.reduce_sum(pos_mask)
    neg_n = tf.reduce_sum(neg_mask)
    pos_weights_mean = tf.math.divide_no_nan(
        tf.reduce_sum(pos_diag_weights), pos_n)
    neg_weights_mean = tf.math.divide_no_nan(
        tf.reduce_sum(neg_diag_weights), neg_n)
    pos_weights_square_mean = tf.math.divide_no_nan(
        tf.reduce_sum(pos_diag_weights * pos_diag_weights), pos_n)
    neg_weights_square_mean = tf.math.divide_no_nan(
        tf.reduce_sum(neg_diag_weights * neg_diag_weights), neg_n)

    pos_var_of_mean = tf.math.divide_no_nan(
        pos_var * pos_weights_square_mean,
        pos_n * pos_weights_mean * pos_weights_mean)

    neg_var_of_mean = tf.math.divide_no_nan(
        neg_var * neg_weights_square_mean,
        neg_n * neg_weights_mean * neg_weights_mean)

    return pos_var_of_mean, neg_var_of_mean

  def call(self,
           membership: types.TensorType,
           predictions: types.TensorType,
           sample_weight: Optional[types.TensorType] = None):
    """Computes Adjusted MMD the loss value."""
    predictions_kernel, normed_weights, pos_mask, neg_mask = self._preprocess(
        membership, predictions, sample_weight)

    pos_mean, neg_mean, pos_neg_mean = self._calculate_mean(
        predictions_kernel, normed_weights, pos_mask, neg_mask)

    # Calculate variance of mean
    pos_var_of_mean, neg_var_of_mean = self._calculate_var_of_mean(
        predictions_kernel, normed_weights, pos_mask, neg_mask, pos_mean,
        neg_mean)

    var = tf.stop_gradient(pos_var_of_mean + neg_var_of_mean) + _EPSILON

    # Adjusted MMD loss
    loss = pos_mean - 2 * pos_neg_mean + neg_mean
    adjusted_loss = tf.math.divide_no_nan(loss, var)

    return adjusted_loss
