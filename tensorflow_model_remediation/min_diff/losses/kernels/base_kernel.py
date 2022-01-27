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

"""Implementation of MinDiffKernel base class."""

import abc
from typing import Optional

import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.common import types


class MinDiffKernel(abc.ABC):

  # pyformat: disable
  """MinDiffKernel abstract base class.

  Arguments:
    tile_input: Boolean indicating whether to tile inputs before computing the
      kernel (see below for details).

  To be implemented by subclasses:

  - `call()`: contains the logic for the kernel tensor calculation.

  Example subclass Implementation:

  ```
  class GuassKernel(MinDiffKernel):

    def call(x, y):
      return tf.exp(-tf.reduce_sum(tf.square(x - y), axis=2) / 0.01)
  ```

  "Tiling" is a way of expanding the rank of the input tensors so that their
  dimensions work for the operations we need.

  If `x` and `y` are of rank `[N, D]` and `[M, D]` respectively, tiling expands
  them to be: `[N, ?, D]` and `[?, M, D]` where `tf` broadcasting will ensure
  that the operations between them work.
  """
  # pyformat: enable

  def __init__(self, tile_input: bool = True):
    self.tile_input = tile_input

  def __call__(self,
               x: types.TensorType,
               y: Optional[types.TensorType] = None) -> types.TensorType:
    # pyformat: disable
    """Invokes the kernel instance.

    Arguments:
      x: `tf.Tensor` of shape `[N, D]` (if tiling input) or `[N, M, D]` (if not
        tiling input).
      y: Optional `tf.Tensor` of shape `[M, D]` (if tiling input) or `[N, M, D]`
        (if not tiling input).

    If `y` is `None`, it is set to be the same as `x`:

    ```
    if y is None:
      y = x
    ```

    Inputs are tiled if `self.tile_input == True` and left as is otherwise.

    Returns:
      `tf.Tensor` of shape `[N, M]`.
    """
    # pyformat: enable
    if y is None:
      y = x
    if self.tile_input:
      x = x[:, tf.newaxis, :]
      y = y[tf.newaxis, :, :]
    return self.call(x, y)

  @abc.abstractmethod
  @docs.do_not_doc_in_subclasses
  def call(self, x: types.TensorType, y: types.TensorType):
    # pyformat: disable
    """Invokes the `MinDiffKernel` instance.

    Arguments:
      x: `tf.Tensor` of shape `[N, M, D]`.
      y: `tf.Tensor` of shape `[N, M, D]`.

    This method contains the logic for computing the kernel. It must be
    implemented by subclasses.

    Note: This method should not be called directly. To call a kernel on inputs,
    always use the `__call__` method, i.e. `kernel(x, y)`, which relies on the
    `call` method internally.

    Returns:
      `tf.Tensor` of shape `[N, M]`.
    """
    # pyformat: enable
    raise NotImplementedError('Must be implemented in subclasses')

  @docs.do_not_doc_in_subclasses
  def get_config(self):
    """Creates a config dictionary for the `MinDiffKernel` instance.

    Any subclass with additional attributes will need to override this method.
    When doing so, users will mostly likely want to first call `super`.

    Returns:
      A config dictionary for the `MinDiffKernel` isinstance.
    """
    return {'tile_input': self.tile_input}

  @classmethod
  @docs.do_not_doc_in_subclasses
  def from_config(cls, config):

    """Creates a `MinDiffKernel` instance fron the config.

    Any subclass with additional attributes or a different initialization
    signature will need to override this method or `get_config`.

    Returns:
      A new `MinDiffKernel` instance corresponding to `config`.
    """
    return cls(**config)
