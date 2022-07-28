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

"""Implementation of Pairwise Cosine Loss."""

from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.counterfactual.losses import base_loss


@tf.keras.utils.register_keras_serializable()
class PairwiseCosineLoss(base_loss.CounterfactualLoss):

  """Pairwise cosine loss between the original and counterfactual.

  Arguments:
    name: Name used for logging and tracking. Defaults to
      `'pairwise_cosine_loss'`.
    global_batch_size: Global batch size for training. This argument will be
       used to reduce the loss across replicas when using
       `tf.distribution.strategy`.
  """
  # pyformat: enable

  def __init__(self,
               name: Optional[str] = None,
               global_batch_size: Optional[int] = None):
    """Initialize an instance of Pairwise Cosine Loss."""
    super(PairwiseCosineLoss, self).__init__(
        name=name or "pairwise_cosine_loss",
        global_batch_size=global_batch_size)

  def call(self, original: types.TensorType,
           counterfactual: types.TensorType) -> types.TensorType:
    """Computes the cosine distance between the original and counterfactual.

    Arguments:
      original:  The predictions from the original example values. shape =
        `[batch_size, d0, .. dN]` with `Tensor` of type `float32` or `float64`.
        Required.
      counterfactual: The predictions from the counterfactual examples. shape =
        `[batch_size, d0, .. dN]` with `Tensor` of the same type and shape as
        `original`. Required.
    Returns:
     Computed cosine distance between original and counterfactual examples per
     example in the batch shape = `[batch_size]`
    """

    cosine = tf.keras.losses.CosineSimilarity(
        reduction=tf.keras.losses.Reduction.NONE)
    return cosine(original, counterfactual)
