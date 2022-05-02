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

"""Model module for Counterfactual Keras integration.

This Module provides the implementation of a CounterfactualModel, a Model that
delegates its call method to another Model and adds a
`counterfactual_loss` during training and optionally during evaluation.
"""

import tensorflow as tf

from tensorflow_model_remediation.common import docs
from tensorflow_model_remediation.counterfactual import losses
from tensorflow_model_remediation.counterfactual.keras import utils
from tensorflow_model_remediation.counterfactual.keras.utils import structure_utils
from tensorflow_model_remediation.counterfactual.losses import loss_utils


@tf.keras.utils.register_keras_serializable()
class CounterfactualModel(tf.keras.Model):

  """Model that adds a Counterfactual loss component to another model during training.

  Inherits from: `tf.keras.Model`


  Arguments:
    original_model: Instance of `tf.keras.Model` that will be trained with the
      additional `counterfactual_loss`.
    loss: `dict` or single element of string(s) (name of loss) or
      `counterfactual.losses.CounterfactualLoss` instance that will be used to
      calculate the `counterfactual_loss`. Defaults to PairwiseMSELoss.
    loss_weight: `dict` of scalars or single scalar applied to the
      `counterfactual_loss` before being included in training. Defaults to 1.0.
    **kwargs: Named parameters that will be passed directly to the base
      class' `__init__` function.

  `CounterfactualModel` wraps the model passed in, `original_model`, and adds a
  component to the loss during training and optionally during evaluation.

  ### <a id=constructing_counterfactualmodel></a>Construction

  There are two ways to construct a `CounterfactualModel` instance:

  1 - Directly wrap your model with `CounterfactualModel`. This is the simplest
  usage and is most likely what you will want to use (unless your original model
  has some custom implementations that need to be taken into account).

  ```
  import tensorflow as tf

  model = tf.keras.Sequential([...])

  model = CounterfactualModel(model, ...)
  ```

  In this case, all methods other than the ones listed below will use the
  default implementations of `tf.keras.Model`.

  If you are in this use case, the next section is not relevant to you and you
  skip to the section on [usage](#using_counterfactualmodel).


  2 - Subclassing `CounterfactualModel` to integrate custom implementations.
  This will likely be needed if the original_model is itself a customized
  subclass of `tf.keras.Model`. If that is the case and you want to preserve the
  custom implementations, you can create a new custom class that inherits first
  from `CounterfactualModel` and second from your custom class.

  ```
  import tensorflow as tf

  class CustomSequential(tf.keras.Sequential):

    def train_step(self, data):
      print("In a custom train_step!")
      super().train_step(data)

  class CustomCounterfactualModel(CounterfactualModel, CustomSequential):
    pass  # No additional implementation is required.

  model = CustomSequential([...])

  model = CustomCounterfactualModel(model, ...)  # This will use the custom
                                                 # train_step.
  ```

  If you need to customize methods defined by `CounterfactualModel`, then you
  can create a direct subclass and override whatever is needed.

  ```
  import tensorflow as tf

  class CustomCounterfactualModel(CounterfactualModel):

    def update_metrics(self, inputs, ...):
      print("In a custom CounterfactualModel method!")
      super().update_metrics(inputs, ...)

  model = tf.keras.Sequential([...])

  model = CounterfactualModel(model, ...)  # This will use the custom
                                           # update_metrics method.
  ```

  ### <a id=using_counterfactualmodel></a>Usage

  Once you have created an instance of `CounterfactualModel`, it can be used
  almost exactly the same way as the model it wraps. The main two exceptions to
  this are:

  - During training, the inputs must include `counterfactual_data`, see
    `CounterfactualModel.compute_counterfactual_loss` for details.
  - Saving and loading a model can have slightly different behavior if you are
    subclassing `CounterfactualModel`. See `CounterfactualModel.save` and
    `CounterfactualModel.save_original_model` for details.

  Optionally, inputs containing `counterfactual_data` can be passed in to
  `evaluate` and `predict`. For the former, this will result in the
  `counterfactual_loss` appearing in the metrics. For `predict` this
  should have no visible effect.
  """

  def __init__(self,
               original_model: tf.keras.Model,
               loss=losses.PairwiseMSELoss(),
               loss_weight=1.0,
               **kwargs):

    """Initializes a CounterfactualModel instance.
    """
    # Roundabout way of accessing the Functional class.
    functional_class = tf.keras.Sequential.__bases__[0]
    # We need to handle a special case where a custom CounterfactualModel class
    # is created that is also a subclass of the Functional class. In this case,
    # we need to make sure that args match what the Functional.__init__ requires
    # (i.e. `inputs` and `outputs` args) and that the rest of the
    # Functional.__init__ method is skipped (supported by passing in
    # `skip_init=True`).
    # This requires any __init__ methods to not do input validation and to
    # pass through `skip_init`.
    if (isinstance(self, functional_class) and
        not isinstance(self, tf.keras.Sequential)):
      try:
        super(CounterfactualModel, self).__init__(
            inputs=None, outputs=None, skip_init=True, **kwargs)
        tf.keras.Model.__init__(self, **kwargs)
      except Exception as e:
        raise type(e)(
            "There was a problem initializing the CounterfactualModel subclass "
            "instance. This was likely caused by:\n"
            "  - The kwargs that were passed in were not valid according to "
            "tf.keras.Model or a base of your custom Model.\n"
            "  - Some args validation or requirement in your custom Model "
            "__init__ method is too strict.\n"
            "  - Your Model subclass is not passing through **kwargs (in "
            "particular `skip_init`) to the super().__init__ invocation.\n"
            "To fix this, either fix the args, loosen the requirements, or "
            "make sure to pass **kwargs to calls with super. If this is not "
            "possible, you may need to integrate Counterfactual without using "
            "CounterfactualModel.\n"
            f"Error raised: {e}")
    else:
      try:
        super(CounterfactualModel, self).__init__(**kwargs)
      except Exception as e:
        raise type(e)(
            "There was a problem initializing the CounterfactualModel "
            "instance. This was likely caused by the kwargs that were passed "
            "in not being valid according to tf.keras.Model.\n"
            f"Error raised: {e}")

    # Set _auto_track_sub_layers to true to ensure we track the
    # original_model and Counterfactual layers.
    self._auto_track_sub_layers = True  # Track sub layers.
    self.built = True  # This Model is built, original_model may or may not be.
    # Masking, if any, is taken care of by original_model.
    self._supports_masking = False
    # Clear input_spec in case there is one. We cannot make any strong
    # assertions because `counterfactual_data` may or may not be included and
    # can have different shapes since weight is optional.
    self.input_spec = None

    self._original_model = original_model

    self._counterfactual_losses = tf.nest.map_structure(loss_utils._get_loss,
                                                        loss)
    self._counterfactual_loss_weights = _conform_weights_to_losses(
        self._counterfactual_losses, loss_weight)
    self._counterfactual_loss_metrics = []
    self._counterfactual_loss_metrics = _create_unique_metrics(
        self._counterfactual_losses, self._original_model.metrics)
    self._total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
    self._original_loss_metric = tf.keras.metrics.Mean(name="original_loss")

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    all_metrics = [
        self._total_loss_metric, self._counterfactual_loss_metrics,
        self._original_loss_metric
    ]
    if self.compiled_metrics is not None:
      all_metrics.extend(self.compiled_metrics.metrics)
    return all_metrics

  @property
  def original_model(self):
    """`tf.keras.Model` to be trained with the additional `counterfactual_loss`.

    Inference and evaluation will also come from the results this model
    provides.
    """
    return self._original_model

  def compute_total_loss(self, y, y_pred, y_pred_original,
                         y_pred_counterfactual, sample_weight,
                         cf_sample_weight):
    compiled_loss = self.compiled_loss(y, y_pred, sample_weight)
    total_loss = compiled_loss

    counterfactual_loss = None
    if y_pred_counterfactual is not None:
      # Compiled counterfactual losses.
      counterfactual_loss = self.compute_counterfactual_loss(
          y_pred_original, y_pred_counterfactual, cf_sample_weight)
      total_loss += counterfactual_loss
    return total_loss, counterfactual_loss, compiled_loss

  def compute_counterfactual_loss(self, original_predictions,
                                  counterfactual_predictions,
                                  counterfactual_sample_weight):
    """Computes `counterfactual_loss`(es) corresponding to `counterfactual_data`.

    Arguments:
      original_predictions: Predictions on original data.
      counterfactual_predictions: Predictions of a model on counterfactual data.
      counterfactual_sample_weight: Per sample weight to scale counterfactual
        loss.

    Returns:
      Scalar (if only one) or list of `counterfactual_loss` values calculated
        from `counterfactual_data`.
    """


    # Flatten everything and calculate counterfactual_loss for each
    # application.
    flat_losses = structure_utils._flatten_counterfactual_structure(
        self._counterfactual_losses)
    flat_weights = structure_utils._flatten_counterfactual_structure(
        self._counterfactual_loss_weights)
    counterfactual_losses = [
        self._compute_single_counterfactual_loss(
            original_predictions, counterfactual_predictions,
            counterfactual_sample_weight, loss, weight)
        for loss, weight in zip(flat_losses, flat_weights)
    ]
    # If there is only one application return a scalar rather than a list.
    if len(counterfactual_losses) == 1:
      counterfactual_losses = counterfactual_losses[0]
    return counterfactual_losses

  def _compute_single_counterfactual_loss(self, original_predictions,
                                          counterfactual_predictions,
                                          counterfactual_sample_weight, loss_fn,
                                          loss_weight):

    """Computes a single `counterfactual_loss` given a loss, weight, and data.

    This will be called for each application of Counterfactual. See
    `CounterfactualModel.compute_counterfactual_loss` for details.
    """
    counterfactual_loss = loss_weight * loss_fn(
        original=original_predictions,
        counterfactual=counterfactual_predictions,
        sample_weight=counterfactual_sample_weight)
    return counterfactual_loss

  @docs.do_not_doc_in_subclasses
  def train_step(self, data):

    """The logic for one evaluation step.

    Has the exact same behavior as `tf.keras.Model.train_step` with the one
    exception that it adds the 'counterfactual_loss' per step.
    """
    if not isinstance(data, utils.CounterfactualPackedInputs):
      raise ValueError(
          "Training data must be an instance of CounterfactualPackedInputs. "
          f"Received: {data}")
    x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
        data.original_input)
    original_x, cf_x, cf_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(data.counterfactual_data))

    with tf.GradientTape() as tape:
      y_pred = self.original_model(x)
      y_pred_original = self.original_model(original_x)
      y_pred_counterfactual = self.original_model(
          cf_x) if cf_x is not None else None
      total_loss, counterfactual_loss, compiled_loss = self.compute_total_loss(
          y, y_pred, y_pred_original, y_pred_counterfactual, sample_weight,
          cf_sample_weight)

    # Compute gradients
    trainable_vars = self.original_model.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics.
    self.update_metrics(y, y_pred, sample_weight, total_loss, compiled_loss,
                        counterfactual_loss)
    return {m.name: m.result() for m in self.metrics}

  @docs.do_not_doc_in_subclasses
  def call(self, inputs, *args, **kwargs):
    return self.original_model(inputs, *args, **kwargs)

  @docs.do_not_doc_in_subclasses
  def test_step(self, data):

    """The logic for one evaluation step.

    Has the exact same behavior as `tf.keras.Model.test_step` with the one
    exception that it removes the 'counterfactual_loss' metric(s) if
    `counterfactual_data` is not available.
    """
    if isinstance(data, utils.CounterfactualPackedInputs):
      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
          data.original_input)
      original_x, cf_x, cf_sample_weight = (
          tf.keras.utils.unpack_x_y_sample_weight(data.counterfactual_data))
      y_pred = self.original_model(x)
      y_pred_original = self.original_model(original_x)
      y_pred_counterfactual = self.original_model(cf_x)
    else:
      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
      y_pred = self.original_model(x)

      # Set Counterfactual metrics to None.
      y_pred_original = None
      y_pred_counterfactual = None
      cf_sample_weight = None

    total_loss, counterfactual_loss, compiled_loss = self.compute_total_loss(
        y, y_pred, y_pred_original, y_pred_counterfactual,
        sample_weight, cf_sample_weight)
    self.update_metrics(y, y_pred, sample_weight, total_loss, compiled_loss,
                        counterfactual_loss)
    metrics_to_return = []
    for metric in self.metrics:
      if (not isinstance(data, utils.CounterfactualPackedInputs)
         ) and metric == self._counterfactual_loss_metrics:
        continue
      metrics_to_return.append(metric)
    return {m.name: m.result() for m in metrics_to_return}

  @docs.do_not_doc_in_subclasses
  def update_metrics(self, y, y_pred, sample_weight, total_loss, compiled_loss,
                     counterfactual_loss):
    """Updates mean metrics being tracked for Counterfactual losses."""
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    self._total_loss_metric.update_state(total_loss)
    self._original_loss_metric.update_state(compiled_loss)
    if counterfactual_loss is not None:
      self._counterfactual_loss_metrics.update_state(counterfactual_loss)

  # We are overriding this solely to provide complete documentation on the
  # limitations of saving this way as opposed to behavior of normal models.

  def save(self, *args, **kwargs):

    """Exports the model as described in `tf.keras.Model.save`.

    For subclasses of `CounterfactualModel` that have not been registered as
    Keras objects, this method will likely be what you want to call to continue
    training your model with Counterfactual after having loaded it. If you want
    to use the loaded model purely for inference, you will likely want to use
    `CounterfactualModel.save_original_model` instead.

    Note: A model loaded from the output of
      `UnregisteredCounterfactualModelSubclass.save` is slightly different from
      the original instance in that it will require `counterfactual_data` to be
      included in inputs to all functions, even `CounterfactualModel.evaluate`
      and `CounterfactualModel.predict`.

    The exception noted above for unregistered `CounterfactualModel` subclasses
    is the only difference with `tf.keras.Model.save`. To avoid these subtle
    differences, we strongly recommend registering `CounterfactualModel`
    subclasses as Keras objects. See the documentation of
    `tf.keras.utils.register_keras_serializable` for details.
    """
    return super(CounterfactualModel, self).save(*args, **kwargs)

  def save_original_model(self, *args, **kwargs):

    """Exports the `original_model`.

    This model will be the type of `original_model` and will no longer be able
    to train or evaluate with Counterfactual data.

    Note: Since a model loaded from the output of
    `CounterfactualModel.save_original_model` will be an instance of the same
    type as `original_model`, you will need to rewrap it with
    `CounterfactualModel` if you want to train it more with Counterfactual.
    """
    return self.original_model.save(*args, **kwargs)

  def compile(self, *args, **kwargs):

    """Compile both `self` and `original_model` using the same parameters.

    See `tf.keras.Model.compile` for details.
    """
    self.original_model.compile(*args, **kwargs)
    return super(CounterfactualModel, self).compile(*args, **kwargs)

  @docs.do_not_doc_in_subclasses
  def get_config(self):
    """Creates a config dictionary for the `CounterfactualModel` instance.

    Note: This will ignore anything resulting from the kwargs passed in at
    initialization time or changes made to new attributes added afterwards. If
    this is problematic you will need to subclass CounterfactualModel and
    override this method to account for these.

    Any subclass with additional attributes will need to override this method.
    When doing so, users will mostly likely want to first call `super`.

    Returns:
      A config dictionary for the `CounterfactualModel` isinstance.

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
          "CounterfactualModel cannot create a config because `original_model` "
          "has not implemented get_config() or has an error in its "
          "implementation. \nError raised: {}".format(e))

    # Try super.get_config if implemented. In most cases it will not be.
    try:
      config = super(CounterfactualModel, self).get_config()
    except NotImplementedError:
      config = {}

    config.update({
        "original_model": self._original_model,
        "loss": self._counterfactual_losses,
        "loss_weight": self._counterfactual_loss_weights,
        "name": self.name,
    })
    return {k: v for k, v in config.items() if v is not None}

  @classmethod
  def _deserialize_config(cls, config):

    """Takes a config of attributes and deserializes as needed.

    The `original_model` and `loss` are deserialized using the
    `tf.keras.utils.deserialize_keras_object` function.

    Note: This is a convenience method that assumes that the only elements that
    need additional deserialization are original_model` and `loss`. If this is
    not the case for a given subclass this method (or `from_config`) will need
    to be implemented directly.
    """
    return {k: v for k, v in config.items()}

  @classmethod
  @docs.do_not_doc_in_subclasses
  def from_config(cls, config):

    """Creates a `CounterfactualModel` instance from the config.

    Any subclass with additional attributes or a different initialization
    signature will need to override this method or `get_config`.

    Returns:
      A new `CounterfactualModel` instance corresponding to `config`.
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
  """Create uniquely named Counterfactual metric(s) corresponding to loss parameter."""
  if isinstance(loss, tf.keras.losses.Loss):
    return tf.keras.metrics.Mean(
        _unique_metric_name("counterfactual_loss", existing_metrics))

  counterfactual_metrics = []
  for name in loss.keys():
    counterfactual_metrics.append(
        tf.keras.metrics.Mean(
            _unique_metric_name(name + "_counterfactual_loss",
                                existing_metrics)))
  return tf.nest.pack_sequence_as(loss, counterfactual_metrics)


def _conform_weights_to_losses(loss, loss_weight, default_value=1.0):
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
  structure_utils.validate_counterfactual_structure(loss, struct_name="loss")

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
                       f"structures: \n{e}")

  # At this point, we should be guaranteed that the two structures are dicts if
  # they are valid Counterfactual structures. However, in case they are not, we
  # assert that they are both dicts (this also helps be future proof since it
  # will catch the broken assumption immediately if the validity definition
  # changes).
  # Note: As is, it should be impossible to get to this point. The only way it
  #       would is if this function is called without validating or if the
  #       definition of a valid Counterfactual structure has changed.
  if not (isinstance(loss, dict) and isinstance(loss_weight, dict)):
    raise ValueError(
        "One of `loss` and `loss_weight` is neither a single element nor a "
        "dict. This should never happen if they are valid Counterfactual "
        "structures. If you think this is a valid use case (e.g. if the "
        "definition has changed but this piece of code is out of sync), please "
        "file an issue so we can look at it and make the appropriate fix.")

  # Save copy to not alter the original dict.
  loss_weight = loss_weight.copy()

  # First, we make sure to set defaults for any losses that do not have
  # corresponding weights. Raise an error if there are weights with keys that
  # don't correspond to losses.
  if not set(loss_weight.keys()) <= set(loss.keys()):
    raise ValueError(
        "`loss_weight` contains keys that do not correspond to losses:"
        f"\n\nloss: {loss}\n\nloss_weight: {loss_weight}")

  # Provide defaults for any missing weights.
  for key in loss.keys():
    if key not in loss_weight:
      loss_weight[key] = default_value

  # At this point, we should be guaranteed that the two structures match if they
  # are valid Counterfactual structures. However, in case they are not we assert
  # that they match.
  try:
    tf.nest.assert_same_structure(loss, loss_weight)
  except Exception as e:

    raise ValueError(
        "`loss` and `loss_weight` (potentially with default weights added) "
        f"do not have matching structures: \n{e}")

  return loss_weight
