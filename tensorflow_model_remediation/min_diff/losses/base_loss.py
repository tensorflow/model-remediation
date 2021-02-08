# coding=utf-8
# Copyright 2021 Google LLC.
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

"""Implementation of MinDiffLoss base class."""

import abc
from typing import Optional, Text, Tuple
import dill

import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.common import types
from tensorflow_model_remediation.min_diff.losses.kernels import kernel_utils


class MinDiffLoss(tf.keras.losses.Loss, abc.ABC):
  # pyformat: disable

  """MinDiffLoss abstract base class.

  Inherits from: `tf.keras.losses.Loss`

  Arguments:
    membership_transform: Transform function used on `membership`. If `None` is
      passed in then `membership` is left as is. The function must return a
      `tf.Tensor`.
    predictions_transform: Transform function used on `predictions`. If `None`
      is passed in then `predictions` is left as is. The function must return
      a `tf.Tensor`.
    membership_kernel: String (name of kernel) or
      `min_diff.losses.MinDiffKernel` to be applied on `membership`. If `None`
      is passed in, then `membership` is left untouched when applying kernels.
    predictions_kernel: String (name of kernel) or
      `min_diff.losses.MinDiffKernel` to be applied on `predictions`. If `None`
      is passed in, then `predictions` is left untouched when applying kernels.
    name: Name used for logging and tracking.

  To be implemented by subclasses:

  - `call()`: Contains the logic for loss calculation using
    `membership`, `predictions` and optionally `sample_weight`.

  Example subclass implementation:

  ```
  class MyMinDiffLoss(MinDiffLoss):

    def call(membership, predictions, sample_weight=None):
      loss = ...  # Internal logic to calculate loss.
      return loss
  ```

  A `MinDiffLoss` instance measures the difference in prediction scores
  (typically score distributions) between two groups of examples identified by
  the value in the `membership` column.

  If the predictions between the two groups are indistinguishable, the loss
  should be 0. The more different the two scores are, the higher the loss.
  """
  # pyformat: enable

  def __init__(self,
               membership_transform=None,
               predictions_transform=None,
               membership_kernel=None,
               predictions_kernel=None,
               name: Optional[Text] = None):

    """Initialize `MinDiffLoss` instance.

    Raises:
      ValueError: If a `*_transform` parameter is passed in but is not callable.
      ValueError: If a `*_kernel` parameter has an unrecognized type or value.
    """
    super(MinDiffLoss, self).__init__(
        reduction=tf.keras.losses.Reduction.NONE, name=name)
    # TODO: Consider converting to snake case.
    self.name = name or self.__class__.__name__
    _validate_transform(membership_transform, 'membership_transform')
    self.membership_transform = (membership_transform)
    _validate_transform(predictions_transform, 'predictions_transform')
    self.predictions_transform = predictions_transform

    self.membership_kernel = kernel_utils._get_kernel(membership_kernel,
                                                      'membership_kernel')
    self.predictions_kernel = kernel_utils._get_kernel(predictions_kernel,
                                                       'predictions_kernel')

  def __call__(self,
               membership: types.TensorType,
               predictions: types.TensorType,
               sample_weight: Optional[types.TensorType] = None):
    """Invokes the `MinDiffLoss` instance.

    Args:
      membership: Labels indicating whether examples are part of the sensitive
        group. Shape must be `[batch_size, d0, .. dN]`.
      predictions: Predicted values. Must be the same shape as membership.
      sample_weight: (Optional) acts as a coefficient for the loss. Must be of
        shape [batch_size] or [batch_size, 1].  If None then a tensor of ones
        with the appropriate shape is used.

    Returns:
      Scalar `min_diff_loss`.
    """
    with tf.name_scope(self.name + '_inputs'):
      loss = self.call(membership, predictions, sample_weight)

      scalar_version = (
          tf.summary.scalar
          if tf.executing_eagerly() else tf.compat.v1.summary.scalar)

      # Calculate metrics.
      weights = (
          sample_weight
          if sample_weight is not None else tf.ones_like(membership))
      num_min_diff_examples = tf.math.count_nonzero(weights)
      num_sensitive_group_min_diff_examples = tf.math.count_nonzero(weights *
                                                                    membership)
      num_non_sensitive_group_min_diff_examples = (
          num_min_diff_examples - num_sensitive_group_min_diff_examples)
      scalar_version('sensitive_group_min_diff_examples_count',
                     num_sensitive_group_min_diff_examples)
      scalar_version('non-sensitive_group_min_diff_examples_count',
                     num_non_sensitive_group_min_diff_examples)
      scalar_version('min_diff_examples_count', num_min_diff_examples)
      # The following metric can capture when the model degenerates and all
      # predictions go towards zero or one.
      scalar_version(
          'min_diff_average_prediction',
          tf.math.divide_no_nan(
              tf.reduce_sum(tf.dtypes.cast(weights, tf.float32) * predictions),
              tf.cast(num_min_diff_examples, dtype=tf.float32)))

      return loss

  @docs.doc_private
  @docs.do_not_doc_in_subclasses
  def _preprocess_inputs(
      self,
      membership: types.TensorType,
      predictions: types.TensorType,
      sample_weight: Optional[types.TensorType] = None
  ) -> Tuple[types.TensorType, types.TensorType, types.TensorType]:
    # pyformat: disable
    """Preprocesses inputs by applying transforms and normalizing weights.

    Arguments:
      membership: `membership` column as described in
        `MinDiffLoss.call`.
      predictions: `predictions` tensor as described in `MinDiffLoss.call`.
      sample_weight: `sample_weight` tensor as described in `MinDiffLoss.call`.

    The three inputs are processed in the following way:

    - `membership`: has `self.membership_transform`
      applied to it, if it's present, and is cast to `tf.float32`.
    - `predictions`: has `self.predictions_transform` applied to it, if it's
      present, and is cast to `tf.float32`.
    - `sample_weight`: is validated, cast to `tf.float32`, and normalized. If it
      is `None`, it is set to a normalized tensor of equal weights.

    This method is meant for internal use by subclasses when the instance is
    invoked.

    Returns:
      Tuple of (`membership`, `predictions`, `normed_weights`).

    Raises:
      tf.errors.InvalidArgumentError: if `sample_weight` has negative
        entries.
    """
    # pyformat: enable
    # Transform membership if transform is provided and cast.
    if self.membership_transform is not None:
      membership = self.membership_transform(membership)
    membership = tf.cast(membership, tf.float32)
    # Transform predictions if transform is provided and cast.
    if self.predictions_transform is not None:
      predictions = self.predictions_transform(predictions)
    predictions = tf.cast(predictions, tf.float32)
    # Transform weights.
    shape = [tf.shape(membership)[0], 1]
    if sample_weight is None:
      sample_weight = 1.0
    sample_weight = tf.cast(sample_weight, tf.float32)
    sample_weight += tf.zeros(
        shape, dtype=tf.float32)  # Broadcast to the correct shape.
    sample_weight = tf.cast(sample_weight, tf.float32)
    # Raise error if any individual weights are negative.
    assert_op = tf.debugging.assert_non_negative(
        sample_weight,
        message='`sample_weight` cannot contain any negative weights, given: {}'
        .format(sample_weight))
    with tf.control_dependencies([assert_op]):  # Guarantee assert is run first.
      normed_weights = tf.math.divide_no_nan(sample_weight,
                                             tf.reduce_sum(sample_weight))
    return membership, predictions, normed_weights

  @docs.doc_private
  @docs.do_not_doc_in_subclasses
  def _apply_kernels(
      self, membership: types.TensorType, predictions: types.TensorType
  ) -> Tuple[types.TensorType, types.TensorType]:
    # pyformat: disable
    """Applies `losses.MinDiffKernel` attributes to corresponding inputs.

    Arguments:
      membership: `membership` column as described in `MinDiffLoss.call`.
      predictions: `predictions` tensor as described in `MinDiffLoss.call`.

    In particular, `self.membership_kernel`, if not `None`, will
    be applied to `membership` and `self.predictions_kernel`, if not
    `None`, will be applied to `predictions`.

    ```
    loss = ...  # MinDiffLoss subclass instance
    loss.membership_kernel = min_diff.losses.GaussKernel()
    loss.predictions_kernel = min_diff.losses.LaplaceKernel()

    # Call with test inputs.
    loss._apply_kernels([1, 2, 3], [4, 5, 6])  # (GaussKernel([1, 2, 3]),
                                               #  LaplaceKernel([4, 5, 6]))
    ```


    If `self.*_kernel` is `None`, then the corresponding input is returned
    unchanged.

    ```
    loss = ...  # MinDiffLoss subclass instance
    loss.membership_kernel = None
    loss.predictions_kernel = min_diff.losses.GaussKernel
    # Call with test inputs.
    loss._apply_kernels([1, 2, 3], [4, 5, 6])  # ([1, 2, 3], GaussKernel([4, 5, 6]))

    # With both kernels set to None, _apply_kernels is the identity.
    loss.predictions_kernel = None
    # Call with test inputs.
    loss._apply_kernels([1, 2, 3], [4, 5, 6])  # ([1, 2, 3], [4, 5, 6])
    ```

    This function is meant for internal use by subclasses when the instance is
    invoked.

    Returns:
      Tuple of (`membership_kernel_output`, `predictions_kernel_output`).
    """
    # pyformat: enable
    if self.membership_kernel is not None:
      membership = self.membership_kernel(membership)
    if self.predictions_kernel is not None:
      predictions = self.predictions_kernel(predictions)
    return membership, predictions

  @abc.abstractmethod
  @docs.do_not_doc_in_subclasses
  def call(self,
           membership: types.TensorType,
           predictions: types.TensorType,
           sample_weight: Optional[types.TensorType] = None):
    # pyformat: disable
    """Invokes the `MinDiffLoss` instance.

    Arguments:
      membership: Numerical `Tensor` indicating whether examples are
        part of the sensitive_group. This is often denoted with `1.0` or `0.0`
        for `True` or `False` respectively but the details are determined by the
        subclass implementation. Shape must be `[batch_size, 1]`.
      predictions: `Tensor` of model predictions for examples corresponding to
        those in `membership`.
      sample_weight: `Tensor` of weights per example.

    This method contains the logic for calculating the loss. It must be
    implemented by subclasses.

    Note: Like `tf.keras.losses.Loss.call`, this method should not be called
    directly. To call a loss on inputs, always use the `__call__` method,
    i.e. `loss(...)`, which relies on the `call` method internally.

    Returns:
      Scalar `min_diff_loss`.
    """
    # pyformat: enable
    raise NotImplementedError('Must be implemented in subclass.')

  def _serialize_config(self, config):

    """Takes a config of attributes and serializes transforms and kernels.

    Transforms are serialized using the `dill` module. Kernels are serialized
    using the `tf.keras.utils.serialize_keras_object` function.

    Note: This is a convenience method that assumes transform keys end in
    `transform' and kernel keys end in `kernel`. If this is not the case for a
    given subclass, the serialization (or `get_config`) will need to be
    implemented directly.
    """

    def _serialize_value(key, value):
      if key.endswith('transform'):
        return dill.dumps(value)
      return value  # No transformation applied

    return {k: _serialize_value(k, v) for k, v in config.items()}

  @docs.do_not_doc_in_subclasses
  def get_config(self):
    """Creates a config dictionary for the `MinDiffLoss` instance.

    Any subclass with additional attributes will need to override this method.
    When doing so, users will mostly likely want to first call `super`.

    Returns:
      A config dictionary for the `MinDiffLoss` isinstance.
    """
    config = {
        'membership_transform': self.membership_transform,
        'predictions_transform': self.predictions_transform,
        'membership_kernel': self.membership_kernel,
        'predictions_kernel': self.predictions_kernel,
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

    """Creates a `MinDiffLoss` instance from the config.

    Any subclass with additional attributes or a different initialization
    signature will need to override this method or `get_config`.

    Returns:
      A new `MinDiffLoss` instance corresponding to `config`.
    """
    config = cls._deserialize_config(config)
    return cls(**config)


#### Validation Functions ####


def _validate_transform(transform: types.TensorTransformType,
                        var_name: Text) -> None:
  if transform is None:
    return
  if not callable(transform):
    raise ValueError('`{}` should be a callable instance that can be applied '
                     'to a tensor, given: {}'.format(var_name, transform))
