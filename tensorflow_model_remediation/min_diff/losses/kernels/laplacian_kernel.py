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

"""Implementation of LaplacianKernel for MinDiff."""

import tensorflow as tf

from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel


_EPSILON = 1.0e-8


@tf.keras.utils.register_keras_serializable()
class LaplacianKernel(base_kernel.MinDiffKernel):

  # pyformat: disable
  """Laplacian kernel class.

  Arguments:
    kernel_length: Length (sometimes also called 'width') of the kernel.
      Defaults to `0.1`. This parameter essentially describes how far apart
      points can be and still affect each other.

      The choice for kernel length should be influenced by the average distance
      of inputs. The smaller the distance, the smaller the kernel length likely
      needs to be for best performance. In general, a good first guess is the
      standard deviation of your predictions.

      Note: A kernel length that is too large will result in losing most of the
      kernel's non-linearity making it much less effective. A kernel length
      that is too small will make the kernel highly sensitive to input noise
      potentially leading to unstable results.
    **kwargs: Named parameters that will be passed directly to the base
      class' `__init__` function.

  See [paper](https://arxiv.org/abs/1910.11779) for reference on how it can be
  used in MinDiff.
  """
  # pyformat: enable

  def __init__(self, kernel_length: complex = 0.1, **kwargs):
    super(LaplacianKernel, self).__init__(**kwargs)
    self.kernel_length = kernel_length

  def call(self, x: types.TensorType, y: types.TensorType) -> types.TensorType:
    """Computes the Laplacian kernel."""
    # Epsilon is used to avoid non defined gradients.
    return tf.exp(-tf.norm(x - y + _EPSILON, axis=2) / self.kernel_length)

  def get_config(self):
    """Returns the config dictionary for the LaplacianKernel instance."""
    config = super(LaplacianKernel, self).get_config()
    config.update({"kernel_length": self.kernel_length})
    return config
