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

"""Model module for MinDiff Keras integration.

This Module provides the implementation of a MinDiffModel, a Model that
delegates its call method to another Model and adds a `min_diff_loss`
during training and optionally during evaluation.
"""

import dill

import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.min_diff.keras import utils
from tensorflow_model_remediation.min_diff.losses import loss_utils


@tf.keras.utils.register_keras_serializable()
class MinDiffModel(tf.keras.Model):
  # pyformat: disable

  """Model that adds a loss component to another model during training.

  Inherits from: `tf.keras.Model`

  Arguments:
    original_model: Instance of `tf.keras.Model` that will be trained with the
      additional `min_diff_loss`.
    loss: String (name of loss) or `min_diff.losses.MinDiffLoss` instance that
      will be used to calculate the `min_diff_loss`.
    loss_weight: Scalar applied to the `min_diff_loss` before being included
      in training.
    predictions_transform: Optional if the output of `original_model` is a
      `tf.Tensor`. Function that transforms the output of `original_model` after
      it is called on MinDiff examples. The resulting predictions tensor is
      what will be passed in to the `losses.MinDiffLoss`.
    **kwargs: Named parameters that will be passed directly to the base
      class' `__init__` function.

  `MinDiffModel` wraps the model passed in, `original_model`, and adds a
  component to the loss during training and optionally during evaluation.

  ### <a id=constructing_mindiffmodel></a>Construction

  There are two ways to construct a `MinDiffModel` instance, the first is the
  simplest and the most common:

  1 - Directly wrap your model with `MinDiffModel`. This is the simplest usage
  and is most likely what you will want to use (unless your original model has
  some custom implementations that need to be taken into account).

  ```
  import tensorflow as tf

  model = tf.keras.Sequential([...])

  model = MinDiffModel(model, ...)
  ```

  In this case, all methods other than the ones listed below will use the
  default implementations of `tf.keras.Model`.

  If you are in this use case, the next section is not relevant to you and you
  skip to the section on [usage](#using_mindiffmodel).


  2 - Subclassing `MinDiffModel` to integrate custom implementations. This will
  likely be needed if the original_model is itself a customized subclass of
  `tf.keras.Model`. If that is the case and you want to preserve the custom
  implementations, you can create a new custom class that inherits first from
  `MinDiffModel` and second from your custom class.

  ```
  import tensorflow as tf

  class CustomSequential(tf.keras.Sequential):

    def train_step(self, data):
      print("In a custom train_step!")
      super().train_step(data)

  class CustomMinDiffModel(MinDiffModel, CustomSequential):
    pass  # No additional implementation is required.

  model = CustomSequential([...])

  model = CustomMinDiffModel(model, ...)  # This will use the custom train_step.
  ```

  If you need to customize methods defined by `MinDiffModel`, then you can
  create a direct subclass and override whatever is needed.

  ```
  import tensorflow as tf

  class CustomMinDiffModel(MinDiffModel):

    def unpack_min_diff_data(self, inputs):
      print("In a custom MinDiffModel method!")
      super().unpack_min_diff_data(inputs)

  model = tf.keras.Sequential([...])

  model = CustomMinDiffModel(model, ...)  # This will use the custom
                                          # unpack_min_diff_data method.
  ```

  ### <a id=using_mindiffmodel></a>Usage

  Once you have created an instance of `MinDiffModel`, it can be used almost
  exactly the same way as the model it wraps. The main two exceptions to this
  are:

  - During training, the inputs must include `min_diff_data`, see
    `MinDiffModel.compute_min_diff_loss` for details.
  - Saving and loading a model has slightly different behavior. See
    `MinDiffModel.save` and `MinDiffModel.save_original_model` for details.

  Optionally, inputs containing `min_diff_data` can be passed in to `evaluate`
  and `predict`. For the former, this will result in the `min_diff_loss`
  appearing in the metrics. For `predict` this should have no visible effect.
  """
  # pyformat: enable

  def __init__(self,
               original_model: tf.keras.Model,
               loss,
               loss_weight: complex = 1.0,
               predictions_transform=None,
               **kwargs):

    """Initializes a MinDiffModel instance.

    Raises:
      ValueError: If `predictions_transform` is passed in but not callable.
    """

    super(MinDiffModel, self).__init__(**kwargs)
    # Set _auto_track_sub_layers to true to ensure we track the
    # original_model and MinDiff layers.
    self._auto_track_sub_layers = True  # Track sub layers.
    self.built = True  # This Model is built, original_model may or may not be.

    self._original_model = original_model
    self._loss = loss_utils._get_loss(loss)
    self._loss_weight = loss_weight
    self._min_diff_loss_metric = tf.keras.metrics.Mean("min_diff_loss")

    if (predictions_transform is not None and
        not callable(predictions_transform)):
      raise ValueError("`predictions_transform` must be callable if passed "
                       "in, given: {}".format(predictions_transform))
    self._predictions_transform = predictions_transform

    # Clear input_spec in case there is one. We cannot make any strong
    # assertions because `min_diff_data` may or may not be included and can
    # have different shapes since weight is optional.
    self.input_spec = None

  @property
  def predictions_transform(self):
    """Function to be applied on MinDiff predictions before calculating loss.

    MinDiff predictions are the output of `original_model` on the MinDiff
    examples (see `compute_min_diff_loss` for details). These might not
    initially be a `tf.Tensor`, for example if the model is multi-output. If
    this is the case, the predictions need to be converted into a `tf.Tensor`.

    This can be done by selecting one of the outputs or by combining them in
    some way.

    ```
    # Pick out a specific output to use for MinDiff.
    transform = lambda predictions: predictions["output2"]

    model = MinDiffModel(..., predictions_transform=transform)

    # test data imitating multi_output predictions
    test_predictions = {
      "output1": [1, 2, 3],
      "output2": [4, 5, 6],
    }
    model.predictions_transform(test_predictions)  # [4, 5, 6]
    ```

    If no `predictions_transform` parameter is passed in (or `None` is used),
    then it will default to the identity.

    ```
    model = MinDiffModel(..., predictions_transform=None)

    model.predictions_transform([1, 2, 3])  # [1, 2, 3]
    ```

    The result of applying `predictions_transform` on the MinDiff predictions
    must be a `tf.Tensor`. The `min_diff_loss` will be calculated on these
    results.
    """
    if self._predictions_transform is None:
      return lambda predictions: predictions
    return self._predictions_transform

  @property
  def original_model(self):
    """`tf.keras.Model` to be trained with the additional `min_diff_loss` term.

    Inference and evaluation will also come from the results this model
    provides.
    """
    return self._original_model

  def _call_original_model(self, inputs, training=None, mask=None):
    """Calls the original model with appropriate args."""

    arg_tuples = [("training", training,
                   self.original_model._expects_training_arg),
                  ("mask", mask, self.original_model._expects_mask_arg)]

    kwargs = {name: value for name, value, expected in arg_tuples if expected}
    return self.original_model(inputs, **kwargs)

  def unpack_original_inputs(self, inputs):
    # pyformat: disable
    """Extracts original_inputs from `inputs`.

    Arguments:
      inputs: `inputs` as described in `MinDiffModel.call`.

    Identifies whether `min_diff_data` is included in `inputs`. If it is, then
    what is returned is the component that is only meant to be used in the call
    to `original_model`.

    ```
    model = ...  # MinDiffModel.

    inputs = ...  # Batch containing `min_diff_data`

    # Extracts component that is only meant to be passed to `original_model`.
    original_inputs = model.unpack_original_inputs(inputs)
    ```

    If `min_diff_data` is not included, then `inputs` is returned directly.

    ```
    model = ...  # MinDiffModel.

    # Test batch without `min_diff_data` (i.e. just passing in a simple array)
    print(model.unpack_original_inputs([1, 2, 3]))  # [1, 2, 3]
    ```

    The default implementation is a pure wrapper around
    `min_diff.keras.utils.unpack_original_inputs`. See there for implementation
    details.

    Returns:
      Inputs to be used in the call to `original_model`.

    """
    # pyformat: enable
    return utils.unpack_original_inputs(inputs)

  def unpack_min_diff_data(self, inputs):
    # pyformat: disable
    """Extracts `min_diff_data` from `inputs` if present or returns `None`.

    Arguments:
      inputs: `inputs` as described in `MinDiffModel.call`.


    Identifies whether `min_diff_data` is included in `inputs` and returns
    `min_diff_data` if it is.

    ```
    model = ...  # MinDiffModel.

    inputs = ...  # Batch containing `min_diff_data`

    min_diff_data = model.unpack_min_diff_data(inputs)
    ```

    If `min_diff_data` is not included, then `None` is returned.

    ```
    model = ...  # MinDiffModel.

    # Test batch without `min_diff_data` (i.e. just passing in a simple array)
    print(model.unpack_min_diff_data([1, 2, 3]))  # None
    ```

    The default implementation is a pure wrapper around
    `min_diff.keras.utils.unpack_min_diff_data`. See there for implementation
    details.


    Returns:
      `min_diff_data` to be passed to `MinDiffModel.compute_min_diff_loss` if
      present or `None` otherwise.
    """
    # pyformat: enable
    return utils.unpack_min_diff_data(inputs)

  def compute_min_diff_loss(self, min_diff_data, training=None, mask=None):
    # pyformat: disable
    """Computes and returns the `min_diff_loss` corresponding to `min_diff_data`.

    Arguments:
      min_diff_data: Tuple of length 2 or 3 as described below.
      training: Boolean indicating whether to run in training or inference mode.
        See `tf.keras.Model.call` for details.
      mask: Mask or list of masks as described in `tf.keras.Model.call`.


    Like the input requirements described in `tf.keras.Model.fit`,
    `min_diff_data` must be a tuple of length 2 or 3. The tuple will be unpacked
    using the standard `tf.keras.utils.unpack_x_y_sample_weight` function:

    ```
    min_diff_data = ...  # Single batch of min_diff_data.

    min_diff_x, min_diff_membership, min_diff_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(min_diff_data))
    ```
    The components are defined as follows:

    - `min_diff_x`: inputs to `original_model` to get the corresponding MinDiff
      predictions.
    - `min_diff_membership`: numerical [batch_size, 1] `Tensor` indicating which
      group each example comes from (marked as `0.0` or `1.0`).
    - `min_diff_sample_weight`: Optional weight `Tensor`. The weights will be
      applied to the examples during the `min_diff_loss` calculation.

    The `min_diff_loss` is ultimately calculated from the MinDiff
    predictions which are evaluated in the following way:

    ```
    ...  # In compute_min_diff_loss call.

    min_diff_x = ...  # Single batch of MinDiff examples.

    # Get predictions for MinDiff examples.
    min_diff_predictions = self.original_model(min_diff_x, training=training)
    # Transform the predictions if needed. By default this is the identity.
    min_diff_predictions = self.predictions_transform(min_diff_predictions)
    ```

    Returns:
      `min_diff_loss` calculated from `min_diff_data`.

    Raises:
      ValueError: If the transformed `min_diff_predictions` is not a
        `tf.Tensor`.
    """
    # pyformat: enable
    x, membership, sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(min_diff_data))

    predictions = self._call_original_model(x, training=training, mask=mask)
    # Clear any losses added when calling the original model on the MinDiff
    # examples. The right losses, if any, will be added when the original_model
    # is called on the original inputs.
    self._clear_losses()

    predictions = self.predictions_transform(predictions)
    if not isinstance(predictions, tf.Tensor):
      err_msg = (
          "MinDiff `predictions` meant for calculating the `min_diff_loss`"
          "must be a Tensor, given: {}\n".format(predictions))
      if self._predictions_transform is None:
        err_msg += (
            "This is due to the fact that `original_model` does not return "
            "a Tensor either because it is multi output or because it has some "
            "custom implementation. To handle this, pass in a "
            "`predictions_transform` that converts the result into the tensor "
            "the `min_diff_loss` should be calculated on.")
      else:
        err_msg += ("This is due to the fact that the provided "
                    "`predictions_transform` parameter does not return a "
                    "Tensor when given the output of `original_model`.")
      err_msg += "\nSee `MinDiffModel` for additional documentation."

      raise ValueError(err_msg)

    min_diff_loss = self._loss_weight * self._loss(
        predictions=predictions,
        membership=membership,
        sample_weight=sample_weight)
    self._min_diff_loss_metric.update_state(min_diff_loss)

    return min_diff_loss

  @docs.do_not_doc_in_subclasses
  def call(self, inputs, training=None, mask=None):
    # pyformat: disable
    """Calls `original_model` with optional `min_diff_loss` as regularization loss.

    Args:
      inputs: Inputs to original_model, optionally containing `min_diff_data` as
        described below.
      training: Boolean indicating whether to run in training or inference mode.
        See `tf.keras.Model.call` for details.
      mask: Mask or list of masks as described in `tf.keras.Model.call`.

    Note: Like `tf.keras.Model.call`, this method should not be called directly.
    To call a model on an input, always use the `__call__` method,
    i.e. `model(inputs)`, which relies on the `call` method internally.

    This method should be used the same way as `tf.keras.Model.call`. Depending
    on whether you are in train mode, `inputs` may need to include
    `min_diff_data` (see `MinDiffModel.compute_min_diff_data` for details on
    what form that needs to take).

    - If `training=True`: `inputs` must contain `min_diff_data` (see details
      below).
    - If `training=False`: including `min_diff_data` is optional.

    If present, the `min_diff_loss` is added by calling `self.add_loss` and will
    show up in `self.losses`.

    ```
    model = ...  # MinDiffModel.

    dataset = ...  # Dataset containing min_diff_data.

    for batch in dataset.take(1):
      model(batch, training=True)

    model.losses[0]  # First element will be the min_diff_loss.
    ```

    Including `min_diff_data` in `inputs` implies that
    `MinDiffModel.unpack_original_inputs` and
    `MinDiffModel.unpack_min_diff_data` behave as expected when called on
    `inputs` (see methods for details).

    This condition is satisfied with the default implementations if you use
    `min_diff.keras.utils.pack_min_diff_data` to create the dataset that
    includes `min_diff_data`.


    Returns:
      A `tf.Tensor` or nested structure of `tf.Tensor`s according to the
      behavior `original_model`. See `tf.keras.Model.call` for details.

    Raises:
      ValueError: If `training` is set to `True` but `inputs` does not include
        `min_diff_data`.
    """
    # pyformat: enable
    original_inputs = self.unpack_original_inputs(inputs)
    min_diff_data = self.unpack_min_diff_data(inputs)

    # If training is True, we require min_diff_data to be available.
    if training and min_diff_data is None:
      raise ValueError(
          "call `inputs` must contain MinDiffData during training.")

    # Add min_diff_loss if min_diff_data is available.
    if min_diff_data is not None:
      min_diff_loss = self.compute_min_diff_loss(
          min_diff_data, training=training)
      self.add_loss(min_diff_loss)

    return self._call_original_model(
        original_inputs, training=training, mask=mask)

  @docs.do_not_generate_docs
  def test_step(self, data, *args, **kwargs):

    """The logic for one evaluation step.

    Has the exact same behavior as `tf.keras.Model.test_step` with the one
    exception that it removes the 'min_diff_loss' metric if min_diff_data is not
    available.
    """
    metrics = super(MinDiffModel, self).test_step(data, *args, **kwargs)
    # If there is no min_diff_data, remove the min_diff_loss metric.
    x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
    if (self._min_diff_loss_metric.name in metrics and
        self.unpack_min_diff_data(x) is None):
      del metrics[self._min_diff_loss_metric.name]
    return metrics

  # We are overriding this solely to provide complete documentation on the
  # limitations of saving this way as opposed to behavior of normal models.

  def save(self, *args, **kwargs):

    """Exports the model as described in `tf.keras.Model.save`.

    You may want to use this if you want to continue training your model with
    MinDiff after having loaded it. If you want to use the loaded model purely
    for inference, you will likely want to use
    `MinDiffModel.save_original_model` instead.

    Note: A model loaded from the output of `MinDiffModel.save` is slightly
      different from the original instance in that it will require
      `min_diff_data` to be included in inputs to all functions, even
      `MinDiffModel.evaluate` and `MinDiffModel.predict`.

    Other than the exception noted above, this method has the same behavior as
    `tf.keras.Model.save`.
    """
    return super(MinDiffModel, self).save(*args, **kwargs)

  def save_original_model(self, *args, **kwargs):

    """Exports the `original_model` for inference without `min_diff_data`.

    Saving the `original_model` allows you to load a model and run
    `tf.keras.Model.evaluate` or `tf.keras.Model.predict` without requiring
    `min_diff_data` to be included.

    This is most likely what you will want to use if you want to save your model
    for inference only. Most cases will need to use this method instead of
    `MinDiffModel.save`.

    Note: A model loaded from the output of `MinDiffModel.save_original_model`
    will be an instance of the same type as `original_model`, not
    `MinDiffModel`. This means that if you want to train it more with MinDiff,
    you will need to rewrap it with `MinDiffModel`.
    """
    return self.original_model.save(*args, **kwargs)

  def compile(self, *args, **kwargs):

    """Compile both `self` and `original_model` using the same parameters.

    See `tf.keras.Model.compile` for details.
    """
    self.original_model.compile(*args, **kwargs)
    return super(MinDiffModel, self).compile(*args, **kwargs)

  @docs.do_not_doc_in_subclasses
  def get_config(self):
    """Creates a config dictionary for the `MinDiffModel` instance.

    Note: This will ignore anything resulting from the kwargs passed in at
    initialization time or changes made to new attributes added afterwards. If
    this is problematic you will need to subclass MinDiffModel and override this
    method to account for these.

    Any subclass with additional attributes will need to override this method.
    When doing so, users will mostly likely want to first call `super`.

    Returns:
      A config dictionary for the `MinDiffModel` isinstance.

    Raises:
      Exception: If calling `original_model.get_config()` raises an error. The
        type raised will be the same as that of the original error.
    """

    # Check that original_model.get_config is implemented and raise a helpful
    # error message if not.
    try:
      _ = self._original_model.get_config()
    except Exception as e:
      raise type(e)(
          "MinDiffModel cannot create a config because `original_model` has "
          "not implemented get_config() or has an error in its implementation."
          "\nError raised: {}".format(e))

    # Try super.get_config if implemented. In most cases it will not be.
    try:
      config = super(MinDiffModel, self).get_config()
    except NotImplementedError:
      config = {}

    config.update({
        "original_model": self._original_model,
        "loss": self._loss,
        "loss_weight": self._loss_weight,
        "name": self.name,
    })
    if self._predictions_transform is not None:
      config["predictions_transform"] = dill.dumps(self._predictions_transform)
    return {k: v for k, v in config.items() if v is not None}

  @classmethod
  def _deserialize_config(cls, config):

    """Takes a config of attributes and deserializes as needed.

    Transforms are deserialized using the `dill` module. The `original_model`
    and `loss` are deserialized using the
    `tf.keras.utils.deserialize_keras_object` function.

    Note: This is a convenience method that assumes that the only elements that
    need additional deserialization are `predictions_transform`, original_model`
    and `loss`. If this is not the case for a given subclass this method (or
    `from_config`) will need to be implemented directly.
    """

    def _deserialize_value(key, value):
      if key == "predictions_transform":
        return dill.loads(value)
      return value  # No transformation applied.

    return {k: _deserialize_value(k, v) for k, v in config.items()}

  @classmethod
  @docs.do_not_doc_in_subclasses
  def from_config(cls, config):

    """Creates a `MinDiffModel` instance from the config.

    Any subclass with additional attributes or a different initialization
    signature will need to override this method or `get_config`.

    Returns:
      A new `MinDiffModel` instance corresponding to `config`.
    """
    config = cls._deserialize_config(config)
    return cls(**config)
