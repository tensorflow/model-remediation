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

"""Implementation of the Absolute Correlation Loss."""

from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.min_diff.losses import base_loss


_EPSILON = 1.0e-7


@tf.keras.utils.register_keras_serializable()
class AbsoluteCorrelationLoss(base_loss.MinDiffLoss):

  # pyformat: disable
  """Absolute correlation between predictions on two groups of examples.

  Arguments:
    name: Name used for logging or tracking. Defaults to
      `'absolute_correlation_loss'`.
    enable_summary_histogram: Optional bool indicating if `tf.summary.histogram`
      should be included within the loss. Defaults to True.

  Absolute correlation measures how correlated predictions are with membership
  (regardless of direction). The metric guarantees that the result is 0 if and
  only if the two distributions it is comparing are indistinguishable.

  The `sensitive_group_labels` input is used to determine whether each example
  is part of the sensitive group. This currently only supports hard membership
  of `0.0` or `1.0`.

  For more details, see the [paper](https://arxiv.org/abs/1901.04562).
  """
  # pyformat: enable

  def __init__(self, name: Optional[str] = None,
               enable_summary_histogram: Optional[bool] = True):
    """Initialize Loss."""
    super(AbsoluteCorrelationLoss,
          self).__init__(name=name or 'absolute_correlation_loss',
                         enable_summary_histogram=enable_summary_histogram)

  def call(
      self,
      sensitive_group_labels: types.TensorType,
      y_pred: types.TensorType,
      sample_weight: Optional[types.TensorType] = None) -> types.TensorType:
    """Computes the absolute correlation loss value."""

    sensitive_group_labels, y_pred, normed_weights = self._preprocess_inputs(
        sensitive_group_labels, y_pred, sample_weight)

    weighted_mean_sensitive_group_labels = tf.reduce_sum(normed_weights *
                                                         sensitive_group_labels)
    weighted_mean_y_pred = tf.reduce_sum(normed_weights * y_pred)
    weighted_var_sensitive_group_labels = tf.reduce_sum(
        normed_weights * tf.square(sensitive_group_labels -
                                   weighted_mean_sensitive_group_labels))
    weighted_var_y_pred = tf.reduce_sum(
        normed_weights * tf.square(y_pred - weighted_mean_y_pred))

    weighted_covar = tf.reduce_sum(
        normed_weights *
        (sensitive_group_labels - weighted_mean_sensitive_group_labels) *
        (y_pred - weighted_mean_y_pred))

    # Epsilon is used to avoid non defined gradients.
    corr = tf.math.divide_no_nan(
        weighted_covar,
        tf.sqrt(weighted_var_sensitive_group_labels + _EPSILON) *
        tf.sqrt(weighted_var_y_pred + _EPSILON))

    loss = tf.abs(corr)
    return loss
