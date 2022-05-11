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

"""Implementation of Pairwise Mean Squared Error Loss."""

from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.counterfactual.losses import base_loss


@tf.keras.utils.register_keras_serializable()
class PairwiseMSELoss(base_loss.CounterfactualLoss):

  """Pairwise mean squared error loss between the original and counterfactual.

  Arguments:
    name: Name used for logging and tracking. Defaults to `'pairwise_mse_loss'`.
  """
  # pyformat: enable

  def __init__(self, name: Optional[str] = None):
    """Initialize an instance of PairwiseMSELoss."""
    super(PairwiseMSELoss, self).__init__(name=name or "pairwise_mse_loss")

  def call(self,
           original: types.TensorType,
           counterfactual: types.TensorType,
           sample_weight: Optional[types.TensorType] = None):
    """Computes the mean squared difference value.

    Arguments:
      original:  The predictions from the original example values. shape =
        `[batch_size, d0, .. dN]` with `Tensor` of type `float32` or `float64`.
        Required.
      counterfactual: The predictions from the counterfactual examples. shape =
        `[batch_size, d0, .. dN]` with `Tensor` of the same type and shape as
        `original`. Required.
      sample_weight: (Optional) `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `original` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)

    Returns:
     Computed L2 distance or mean squared difference loss.
    """

    mse = tf.keras.losses.MeanSquaredError()
    return mse(original, counterfactual, sample_weight=sample_weight)
