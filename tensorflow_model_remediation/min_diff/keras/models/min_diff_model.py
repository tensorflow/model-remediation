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

"""Model module for MinDiff Keras integration.

This Module provides the implementation of a MinDiffModel, a Model that
delegates its call method to another Model and adds a `min_diff_loss`
during training and optionally during evaluation.
"""
import inspect
import dill

import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.min_diff.keras import utils
from tensorflow_model_remediation.min_diff.keras.utils import structure_utils
from tensorflow_model_remediation.min_diff.losses import loss_utils


@tf.keras.utils.register_keras_serializable()
class MinDiffModel(tf.keras.Model):
  # pyformat: disable

  """Model that adds one or more loss component(s) to another model during training.

  Inherits from: `tf.keras.Model`

  Arguments:
    original_model: Instance of `tf.keras.Model` that will be trained with the
      additional `min_diff_loss`.
    loss: `dict` or single element of string(s) (name of loss) or
      `min_diff.losses.MinDiffLoss` instance(s) that will be used to calculate
      the `min_diff_loss`(es).
    loss_weight: `dict` of scalars or single scalar applied to the
      `min_diff_loss`(es) before being included in training.
    predictions_transform: Optional if the output of `original_model` is a
      `tf.Tensor`. Function that transforms the output of `original_model` after
      it is called on MinDiff examples. The resulting predictions tensor is
      what will be passed in to the `losses.MinDiffLoss`(es).
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

  ### <a id=multiple_applications></a>Multiple Applications of MinDiff

  It is possible to apply MinDiff multiple times within a single instance of
  `MinDiffModel`. To do so, you can pass in a dictionary of losses where keys
  are the names of each MinDiff application and the values are the names or
  instances of `losses.MinDiffLoss` that will be applied for each respective
  MinDiff application.
  Loss weights can be set as either one value that will be used for all
  applications or with a dictionary that specifies weights for individual
  applications. Weights not specified will default to 1.0.

  ```
  import tensorflow as tf

  model = tf.keras.Sequential([...])

  model = MinDiffModel(model, loss={
    "application1": min_diff.losses.MMDLoss(),  # Loss for first application.
    "application2": min_diff.losses.MMDLoss()   # Loss for second application.
  },
  loss_weight=2.0)  # 2.0 will used as the weight for all applications.
  ```

  A `MinDiffModel` initialized as shown above will expect `min_diff_data` to
  have a structure matching that of `loss` (i.e. a dictionary of inputs with
  keys matching that of `loss`). See `MinDiffModel.compute_min_diff_loss` for
  details.

  ### <a id=using_mindiffmodel></a>Usage

  Once you have created an instance of `MinDiffModel`, it can be used almost
  exactly the same way as the model it wraps. The main two exceptions to this
  are:

  - During training, the inputs must include `min_diff_data`, see
    `MinDiffModel.compute_min_diff_loss` for details.
  - Saving and loading a model can have slightly different behavior if you are
    subclassing `MinDiffModel`. See `MinDiffModel.save` and
    `MinDiffModel.save_original_model` for details.

  Optionally, inputs containing `min_diff_data` can be passed in to `evaluate`
  and `predict`. For the former, this will result in the `min_diff_loss`
  appearing in the metrics. For `predict` this should have no visible effect.
  """
  # pyformat: enable

  def __init__(self,
               original_model: tf.keras.Model,
               loss,
               loss_weight=1.0,
               predictions_transform=None,
               **kwargs):

    """Initializes a MinDiffModel instance.

    Raises:
      ValueError: If `predictions_transform` is passed in but not callable.
    """
    # Roundabout way of accessing the Functional class.
    functional_class = tf.keras.Sequential.__bases__[0]
    # We need to handle a special case where a custom MinDiffModel class is
    # created that is also a subclass of the Functional class. In this case, we
    # need to make sure that args match what the Functional.__init__ requires
    # (i.e. `inputs` and `outputs` args) and that the rest of the
    # Functional.__init__ method is skipped (supported by passing in
    # `skip_init=True`).
    # This requires any __init__ methods to not do input validation and to
    # pass through `skip_init`.
    if (isinstance(self, functional_class) and
        not isinstance(self, tf.keras.Sequential)):
      try:
        super(MinDiffModel, self).__init__(
            inputs=None, outputs=None, skip_init=True, **kwargs)
        tf.keras.Model.__init__(self, **kwargs)
      except Exception as e:
        raise type(e)(
            "There was a problem initializing the MinDiffModel subclass "
            "instance. This was likely caused by:\n"
            "  - The kwargs that were passed in were not valid according to "
            "tf.keras.Model or a base of your custom Model.\n"
            "  - Some args validation or requirement in your custom Model "
            "__init__ method is too strict.\n"
            "  - Your Model subclass is not passing through **kwargs (in "
            "particular `skip_init`) to the super().__init__ invocation.\n"
            "To fix this, either fix the args, loosen the requirements, or "
            "make sure to pass **kwargs to calls with super. If this is not "
            "possible, you may need to integrate MinDiff without using "
            "MinDiffModel.\n"
            "Error raised: {}".format(e))
    else:
      try:
        super(MinDiffModel, self).__init__(**kwargs)
      except Exception as e:
        raise type(e)(
            "There was a problem initializing the MinDiffModel instance. "
            "This was likely caused by the kwargs that were passed in not "
            "being valid according to tf.keras.Model.\n"
            "Error raised: {}".format(e))

    # Set _auto_track_sub_layers to true to ensure we track the
    # original_model and MinDiff layers.
    self._auto_track_sub_layers = True  # Track sub layers.
    self.built = True  # This Model is built, original_model may or may not be.
    # Masking, if any, is taken care of by original_model.
    self._supports_masking = False
    # Clear input_spec in case there is one. We cannot make any strong
    # assertions because `min_diff_data` may or may not be included and can
    # have different shapes since weight is optional.
    self.input_spec = None

    self._original_model = original_model
    structure_utils.validate_min_diff_structure(loss, struct_name="loss")
    self._loss = tf.nest.map_structure(loss_utils._get_loss, loss)
    structure_utils.validate_min_diff_structure(
        loss_weight, struct_name="loss_weight")
    self._loss_weight = _conform_weights_to_losses(
        self._loss, loss_weight, default_value=1.0)
    self._min_diff_loss_metric = _create_unique_metrics(self._loss,
                                                        self.metrics)

    if (predictions_transform is not None and
        not callable(predictions_transform)):
      raise ValueError("`predictions_transform` must be callable if passed "
                       "in, given: {}".format(predictions_transform))
    self._predictions_transform = predictions_transform

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
                   self.original_model._expects_training_arg)]

    # Check if the original model call signature uses "mask" and pass mask to
    # the original model if present.
    if "mask" in inspect.getfullargspec((self.original_model.call)).args:
      arg_tuples.append(("mask", mask, self.original_model._expects_mask_arg))
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
    """Computes `min_diff_loss`(es) corresponding to `min_diff_data`.

    Arguments:
      min_diff_data: Tuple of data or valid MinDiff structure of tuples as
        described below.
      training: Boolean indicating whether to run in training or inference mode.
        See `tf.keras.Model.call` for details.
      mask: Mask or list of masks as described in `tf.keras.Model.call`. These
        will be applied when calling the `original_model`.


    `min_diff_data` must have a structure (or be a single element) matching that
    of the `loss` parameter passed in during initialization. Each element of
    `min_diff_data` (and `loss`) corresponds to one application of MinDiff.

    Like the input requirements described in `tf.keras.Model.fit`, each element
    of `min_diff_data` must be a tuple of length 2 or 3. The tuple will be
    unpacked using the standard `tf.keras.utils.unpack_x_y_sample_weight`
    function:

    ```
    min_diff_data_elem = ...  # Single element from a batch of min_diff_data.

    min_diff_x, min_diff_membership, min_diff_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(min_diff_data_elem))
    ```
    The components are defined as follows:

    - `min_diff_x`: inputs to `original_model` to get the corresponding MinDiff
      predictions.
    - `min_diff_membership`: numerical [batch_size, 1] `Tensor` indicating which
      group each example comes from (marked as `0.0` or `1.0`).
    - `min_diff_sample_weight`: Optional weight `Tensor`. The weights will be
      applied to the examples during the `min_diff_loss` calculation.

    For each application of MinDiff, the `min_diff_loss` is ultimately
    calculated from the MinDiff predictions which are evaluated in the
    following way:

    ```
    ...  # In compute_min_diff_loss call.

    min_diff_x = ...  # Single batch of MinDiff examples.

    # Get predictions for MinDiff examples.
    min_diff_predictions = self.original_model(min_diff_x, training=training)
    # Transform the predictions if needed. By default this is the identity.
    min_diff_predictions = self.predictions_transform(min_diff_predictions)
    ```

    Returns:
      Scalar (if only one) or list of `min_diff_loss` values calculated from
        `min_diff_data`.

    Raises:
      ValueError: If the structure of `min_diff_data` does not match that of the
        `loss` that was passed to the model during initialization.
      ValueError: If the transformed `min_diff_predictions` is not a
        `tf.Tensor`.
    """
    # pyformat: enable

    structure_utils._assert_same_min_diff_structure(min_diff_data, self._loss)

    # Flatten everything and calculate min_diff_loss for each application.
    flat_data = structure_utils._flatten_min_diff_structure(min_diff_data)
    flat_losses = structure_utils._flatten_min_diff_structure(self._loss)
    flat_weights = structure_utils._flatten_min_diff_structure(
        self._loss_weight)
    flat_metrics = structure_utils._flatten_min_diff_structure(
        self._min_diff_loss_metric)
    min_diff_losses = [
        self._compute_single_min_diff_loss(data, loss, weight, metric, training,
                                           mask) for data, loss, weight, metric
        in zip(flat_data, flat_losses, flat_weights, flat_metrics)
    ]
    # If there is only one application return a scalar rather than a list.
    if len(min_diff_losses) == 1:
      min_diff_losses = min_diff_losses[0]

    return min_diff_losses

  def _compute_single_min_diff_loss(self,
                                    min_diff_data,
                                    loss,
                                    loss_weight,
                                    min_diff_loss_metric,
                                    training=None,
                                    mask=None):

    """Computes a single `min_diff_loss` given a loss, weight, and data.

    This will be called for each application of MinDiff. See
    `MinDiffModel.compute_min_diff_loss` for details.
    """
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
          "MinDiff `predictions` meant for calculating the `min_diff_loss` "
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

    min_diff_loss = loss_weight * loss(
        predictions=predictions,
        membership=membership,
        sample_weight=sample_weight)
    min_diff_loss_metric.update_state(min_diff_loss)
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

    model.losses[0]  # First element(s) will be the min_diff_loss(es).
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
      # Add min_diff_loss(es) as regularization loss(es).
      tf.nest.map_structure(self.add_loss, min_diff_loss)

    return self._call_original_model(
        original_inputs, training=training, mask=mask)

  @docs.do_not_generate_docs
  def test_step(self, data, *args, **kwargs):

    """The logic for one evaluation step.

    Has the exact same behavior as `tf.keras.Model.test_step` with the one
    exception that it removes the 'min_diff_loss' metric(s) if `min_diff_data`
    is not available.
    """
    metrics = super(MinDiffModel, self).test_step(data, *args, **kwargs)
    # If there is no min_diff_data, remove the min_diff_loss metric.
    x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
    if self.unpack_min_diff_data(x) is None:
      for metric in tf.nest.flatten(self._min_diff_loss_metric):
        if metric.name in metrics:
          del metrics[metric.name]
    return metrics

  # We are overriding this solely to provide complete documentation on the
  # limitations of saving this way as opposed to behavior of normal models.

  def save(self, *args, **kwargs):

    """Exports the model as described in `tf.keras.Model.save`.

    For subclasses of `MinDiffModel` that have not been registered as Keras
    objects, this method will likely be what you want to call to continue
    training your model with MinDiff after having loaded it. If you want to use
    the loaded model purely for inference, you will likely want to use
    `MinDiffModel.save_original_model` instead.

    Note: A model loaded from the output of
      `UnregisteredMinDiffModelSubclass.save` is slightly different from the
      original instance in that it will require `min_diff_data` to be included
      in inputs to all functions, even `MinDiffModel.evaluate` and
      `MinDiffModel.predict`.

    The exception noted above for unregistered `MinDiffModel` subclasses is the
    only difference with `tf.keras.Model.save`. To avoid these subtle
    differences, we strongly recommend registering `MinDiffModel` subclasses as
    Keras objects. See the documentation of
    `tf.keras.utils.register_keras_serializable` for details.
    """
    return super(MinDiffModel, self).save(*args, **kwargs)

  def save_original_model(self, *args, **kwargs):

    """Exports the `original_model`.

    Exports the `original_model`. When loaded, this model will be the type of
    `original_model` and will no longer be able to train or evaluate with
    MinDiff data.

    Note: Since a model loaded from the output of
    `MinDiffModel.save_original_model` will be an instance of the same type as
    `original_model`, you will need to rewrap it with `MinDiffModel` if you want
    to train it more with MinDiff.
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


def _unique_metric_name(name, existing_metrics):
  """Returns a unique name given the existing metric names."""
  existing_names = set([metric.name for metric in existing_metrics])

  proposed_name = name
  cnt = 1  # Start incrementing with 1.
  # Increment name suffix until the name is unique.
  while proposed_name in existing_names:
    proposed_name = name + "_" + str(cnt)
    cnt += 1

  return proposed_name


def _create_unique_metrics(loss, existing_metrics):
  """Create uniquely named MinDiff metric(s) corresponding to loss parameter."""
  if not isinstance(loss, dict):
    return tf.keras.metrics.Mean(
        _unique_metric_name("min_diff_loss", existing_metrics))

  min_diff_metrics = []
  for name in loss.keys():
    min_diff_metrics.append(
        tf.keras.metrics.Mean(
            _unique_metric_name(name + "_min_diff_loss", existing_metrics)))
  return tf.nest.pack_sequence_as(loss, min_diff_metrics)


def _conform_weights_to_losses(loss, loss_weight, default_value):
  """Conforms weights to match structure of losses.

  Shape weights to match the structure of `loss` if possible. If `loss_weight`
  is a single value, it will be broadcast for all losses. If `loss_weight` is
  `None` or has missing entries, `default_value` will be used.

  Args:
    loss: loss (possible nested) that weights will be conformed to.
    loss_weight: weight that will be conformed to loss structure. If only a
      single value, it will be broadcast for all losses. If `None`, it will be
      replaced by `default_value`.
    default_value: Value used if `loss_weight` is `None` or if some weights are
      missing for certain losses.

  Returns:
    Weight corresponding to `loss` structure.
  """
  # Validate loss (loss_weights will be implicitly validated)
  structure_utils.validate_min_diff_structure(loss, struct_name="loss")

  # If loss_weight is unnested, then broadcast to all values of loss.
  if not tf.nest.is_nested(loss_weight):
    if loss_weight is None:
      loss_weight = default_value
    return tf.nest.map_structure(lambda _: loss_weight, loss)

  # If execution reaches here, then loss_weight is nested (a dict).

  # If loss is not nested, then raise an error (since loss_weight is a nested).
  if not tf.nest.is_nested(loss):
    try:
      tf.nest.assert_same_structure(loss, loss_weight)
    except Exception as e:

      raise ValueError("`loss` and `loss_weight` do not have matching "
                       "structures: \n{}".format(e))

  # At this point, we should be guaranteed that the two structures are dicts if
  # they are valid MinDiff structures. However, in case they are not, we assert
  # that they are both dicts (this also helps be future proof since it will
  # catch the broken assumption immediately if the validity definition changes).
  # Note: As is, it should be impossible to get to this point. The only way it
  #       would is if this function is called without validating or if the
  #       definition of a valid MinDiff structure has changed.
  if not (isinstance(loss, dict) and isinstance(loss_weight, dict)):
    raise ValueError(
        "One of `loss` and `loss_weight` is neither a single element nor a "
        "dict. This should never happen if they are valid MinDiff structures. "
        "If you think this is a valid use case (e.g. if the definition has "
        "changed but this piece of code is out of sync), please file an issue "
        "so we can look at it and make the appropriate fix.")

  # Save copy to not alter the original dict.
  loss_weight = loss_weight.copy()

  # First, we make sure to set defaults for any losses that do not have
  # corresponding weights. Raise an error if there are weights with keys that
  # don't correspond to losses.
  if not set(loss_weight.keys()) <= set(loss.keys()):
    raise ValueError(
        "`loss_weight` contains keys that do not correspond to losses:"
        "\n\nloss: {}\n\nloss_weight: {}".format(loss, loss_weight))

  # Provide defaults for any missing weights.
  for key in loss.keys():
    if key not in loss_weight:
      loss_weight[key] = default_value

  # At this point, we should be guaranteed that the two structures match if they
  # are valid MinDiff structures. However, in case they are not we assert that
  # they match.
  try:
    tf.nest.assert_same_structure(loss, loss_weight)
  except Exception as e:

    raise ValueError(
        "`loss` and `loss_weight` (potentially with default weights added) "
        "do not have matching structures: \n{}".format(e))

  return loss_weight
