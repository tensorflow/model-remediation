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

"""Implementation of CounterfactualLoss base class."""

import abc
import re
from typing import Optional

import dill
import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.common import types


class CounterfactualLoss(tf.keras.losses.Loss, abc.ABC):
  """CounterfactualLoss abstract base class.

  Inherits from: `tf.keras.losses.Loss`

  A `CounterfactualLoss` instance measures the difference in prediction scores
  (typically score distributions) between two groups of examples identified by
  the value in the `counterfactual_weights` column.

  If the predictions between the two groups are indistinguishable, the loss
  should be 0. The greater different between the two scores are, the higher the
  loss.
  """

  def __init__(self, name: Optional[str] = None):
    """Initialize `CounterfactualLoss` instance."""
    super(CounterfactualLoss, self).__init__(
        reduction=tf.keras.losses.Reduction.NONE, name=name)
    self.name = name or _to_snake_case(self.__class__.__name__)

  def __call__(self,
               original: types.TensorType,
               counterfactual: types.TensorType,
               sample_weight: Optional[types.TensorType] = None):
    """Computes Counterfactual loss.

    Args:
      original:  The predictions from the original example values. shape =
        `[batch_size, d0, .. dN]`. `Tensor` of type `float32` or `float64`.
        Required.
      counterfactual: The predictions from the counterfactual examples. shape =
        `[batch_size, d0, .. dN]`. `Tensor` of the same type and shape as
        `original`. Required.
      sample_weight: (Optional) `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector.

    Returns:
      The computed counterfactual loss.

    Raises:
      ValueError: If any of the input arguments are invalid.
      TypeError: If any of the arguments are not of the expected type.
      InvalidArgumentError: If `original`, `counterfactual` or `sample_weight`
        have incompatible shapes.
    """
    with tf.name_scope(self.name + '_inputs'):
      if original is None:
        raise ValueError('Argument `original` must not be None.')
      if counterfactual is None:
        raise ValueError('Argument `counterfactual` must not be None.')
      if original.shape.as_list() != counterfactual.shape.as_list():
        raise ValueError(
            'Incompatible shape {} vs {}|Dimensions must be equal.'.format(
                original.shape.as_list(), counterfactual.shape.as_list()))

      # The true and false arms for tf.cond are not equivalent, so


      return tf.cond(
          tf.equal(tf.size(original),
                   0), lambda: self._calculate_and_summarise_loss(),
          lambda: self._calculate_and_summarise_loss(original, counterfactual,
                                                     sample_weight))

  @abc.abstractmethod
  @docs.do_not_doc_in_subclasses
  def call(self,
           original: types.TensorType,
           target: types.TensorType,
           sample_weight: Optional[types.TensorType] = None):
    # pyformat: disable
    """Invokes the `CounterfactualLoss` instance.

    Arguments:
      original:  The predictions from the original example values.
        shape = `[batch_size, d0, .. dN]`. `Tensor` of type `float32` or
        `float64`. Required.
      target: The predictions from the counterfactual examples.
        shape = `[batch_size, d0, .. dN]`. `Tensor` of the same type and shape
        as `original`. Required.
      sample_weight: (Optional) `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `original` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)

    This method contains the logic for calculating the loss. It must be
    implemented by subclasses.

    Note: Like `tf.keras.losses.Loss.call`, this method should not be called
    directly. To call a loss on inputs, always use the `__call__` method,
    i.e. `loss(...)`, which relies on the `call` method internally.

    Returns:
      Scalar `counterfactual_loss`.

    Raises:
      NotImplementedError: If a subclass of Counterfactual loss is not
        implemented.
    """
    # pyformat: enable
    raise NotImplementedError('Must be implemented in subclass.')

  def _serialize_config(self, config):

    def _serialize_value(key, value):
      if key.endswith('transform'):
        return dill.dumps(value)
      return value  # No transformation applied

    return {k: _serialize_value(k, v) for k, v in config.items()}

  @docs.do_not_doc_in_subclasses
  def get_config(self):
    """Creates a config dictionary for the `CounterfactualKerasLoss` instance.

    Any subclass with additional attributes will need to override this method.
    When doing so, users will mostly likely want to first call `super`.

    Returns:
      A config dictionary for the `CounterfactualKerasLoss` isinstance.
    """
    config = {
        'name': self.name,
    }
    config = {k: v for k, v in config.items() if v is not None}
    return self._serialize_config(config)

  @classmethod
  def _deserialize_config(cls, config):

    """Takes a config of attributes and deserializes transforms and kernels.

    Transforms are deserialized using the `dill` module. Kernels are
    deserialized using the `tf.keras.utils.deserialize_keras_object` function.

    Note: This is a convenience method that assumes transform keys end in
    `transform' and kernel keys end in `kernel`. If this is not the case for a
    given subclass, the deserialization (or `from_config`) will need to be
    implemented directly.
    """

    def _deserialize_value(key, value):
      if key.endswith('transform'):
        return dill.loads(value)
      return value  # No transformation applied

    return {k: _deserialize_value(k, v) for k, v in config.items()}

  @classmethod
  @docs.do_not_doc_in_subclasses
  def from_config(cls, config):

    """Creates a `CounterfactualLoss` instance from the config.

    Any subclass with additional attributes or a different initialization
    signature will need to override this method or `get_config`.

    Returns:
      A new `CounterfactualLoss` instance corresponding to `config`.
    """
    config = cls._deserialize_config(config)
    return cls(**config)

  def _calculate_and_summarise_loss(
      self,
      original: Optional[types.TensorType] = None,
      counterfactual: Optional[types.TensorType] = None,
      sample_weight: Optional[types.TensorType] = None) -> types.TensorType:
    loss = tf.constant(0.0, dtype=tf.dtypes.float32)
    if original is not None and counterfactual is not None:
      batch_size = original.shape[0]
      if sample_weight is not None and tf.is_tensor(sample_weight):
        if sample_weight.shape.as_list() not in [[None], [], [batch_size],
                                                 [batch_size, 1]]:
          raise ValueError(
              'Incompatible `sample_weight` shape {}. Must be scalar or 1D'
              ' tensor of shape [batch_size] i.e, [{}].'.format(
                  sample_weight.shape.as_list(), batch_size))
      loss = self.call(
          _to_dense_tensor(original), _to_dense_tensor(counterfactual),
          sample_weight)
    tf.summary.scalar('counterfactual_loss', loss)
    return loss


# This is the same function as the one used in tf.keras.Layer
def _to_snake_case(name: str) -> str:
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def _to_dense_tensor(t: types.TensorType):
  if isinstance(t, tf.SparseTensor):
    return tf.sparse.to_dense(t)
  return t
