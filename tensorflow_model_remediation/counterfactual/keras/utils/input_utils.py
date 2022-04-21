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

"""Input utils module for Counterfactual Keras integration.

This module provides default implementations for packing and unpacking
Counterfactual data into or from an input dataset.
"""

import collections

from typing import Any, Callable, List, Optional

import tensorflow as tf
from tensorflow_model_remediation.counterfactual.keras.utils import structure_utils


class CounterfactualPackedInputs(
    collections.namedtuple("CounterfactualPackedInputs", [
        "original_x", "original_y", "original_sample_weight",
        "counterfactual_x", "counterfactual_sample_weight"
    ])):
  """Named tuple containing inputs for the original and counterfactual data.

  `CounterfactualModel` default implementations and `utils.(un)pack_*` functions
  use this class to pack and unpack the separate components required for
  Counterfactual and regular training.

  Attributes:
    original_x: Batch of inputs that would originally (i.e. without
      applying Counterfactual) be passed in to a model's `Model.call` method.
      This corresponds to the `x` component described in `tf.keras.Model.fit`.
    original_y: Original `y` of the original dataset.
    original_sample_weight: Original `weight` of the original dataset.
    counterfactual_x: Counterfactual value based on `original_x` that has been
      modified. This value and original_x will be used to calculate
      `CounterfactualLoss`.
    counterfactual_sample_weight: Weight value assigned to `counterfactual_x`.
  """


def pack_counterfactual_data(
    original_dataset: tf.data.Dataset,
    counterfactual_dataset: tf.data.Dataset,
    cf_sample_weight: Optional[complex] = None) -> tf.data.Dataset:
  """Packs `counterfactual_dataset` with the `original_dataset`.

  Arguments:
    original_dataset: `tf.data.Dataset` that was used before applying
      Counterfactual. The output should conform to the format used in
      `tf.keras.Model.fit`.
    counterfactual_dataset: `tf.data.Dataset` or valid Counterfactual structure
      (unnested dict) of `tf.data.Dataset`s containing only examples to be used
      to calculate the `counterfactual_loss`.
    cf_sample_weight: Optional sample weight to add to the Counterfactual
      dataset.

  This function should be used to create the dataset that will be passed to
  `counterfactual.keras.CounterfactualModel` during training and, optionally,
  during evaluation.

  Warning: All input datasets should be batched **before** being passed in.

  Each input dataset must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  Every batch from the returned `tf.data.Dataset` will contain one batch from
  each of the input datasets. Each returned batch will be a tuple of
  `(packed_inputs, original_y, original_sample_weight, counterfactual_x)`
  matching the length of `original_dataset` batches where:

  - `packed_inputs`: is an instance of `utils.CounterfactualPackedInputs`
     containing:

    - `original_x`: `x` component taken directly from the
        `original_dataset` batch.
    - `original_y`: `y` component taken directly from the
        `original_dataset` batch.
    - `original_sample_weight`: `sample_weight` component taken directly from
        the `original_dataset` batch.
    - `counterfactual_x`: Batch dataset of data formed from
      `counterfactual_dataset` (as described in
      `utils.build_counterfactual_dataset`).
    - `counterfactual_sample_weight`: Batch of data formed from taken directly
      from the `sample_weight` of `counterfactual_dataset` or passed in by the
      user and created with `tf.fill`. A user provided parameter will override
      the provided `sample_weight` in `counterfactual_dataset`.

  `counterfactual_data` will be used in
  `counterfactual.keras.CounterfactualModel` when calculating the
  `counterfactual_loss`.

  Returns:
    A `tf.data.Dataset` whose output is a tuple of `CounterfactualPackedInputs`
      that contains (`original_y`, `original_y`, `original_sample_weight`,
      `counterfactual_x`, `counterfactual_sample_weight`) matching the output
      length of `original_dataset`.

  Raises:
    ValueError: If the original dataset and counterfactual dataset do not
      have the same cardinality.
  """
  # Validate original_dataset and counterfactual_dataset structure.
  structure_utils.validate_counterfactual_structure(
      original_dataset,
      struct_name="original_dataset",
      element_type=tf.data.Dataset)
  structure_utils.validate_counterfactual_structure(
      counterfactual_dataset,
      struct_name="counterfactual_dataset",
      element_type=tf.data.Dataset)

  tf.nest.assert_same_structure(
      original_dataset, counterfactual_dataset)

  original_dataset_cardinality = tf.data.experimental.cardinality(
      original_dataset).numpy()
  counterfactual_dataset_cardinality = tf.data.experimental.cardinality(
      counterfactual_dataset).numpy()
  if original_dataset_cardinality != counterfactual_dataset_cardinality:
    raise ValueError(
        "The cardinality of `original_dataset` and `counterfactual_dataset` "
        "are different. There must be a matching counterfactual example for "
        "each value within the original values. \n\nFound:\n"
        f"Original cardinality: {original_dataset_cardinality}\n"
        f"Counterfactual cardinality: {counterfactual_dataset_cardinality}")

  dataset = tf.data.Dataset.zip((original_dataset, counterfactual_dataset))

  def _map_fn(original_batch, counterfactual_batch):
    # Unpack original batch.
    original_x, original_y, original_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(original_batch))
    counterfactual_x, _, counterfactual_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(counterfactual_batch))

    nonlocal cf_sample_weight
    if cf_sample_weight is not None:
      flat_counterfactual_x = tf.nest.flatten(counterfactual_x)
      counterfactual_sample_weight = tf.fill(
          [tf.shape(flat_counterfactual_x[0])[0], 1], cf_sample_weight)

    return CounterfactualPackedInputs(
        original_x=original_x,
        original_y=original_y,
        original_sample_weight=original_sample_weight,
        counterfactual_x=counterfactual_x,
        counterfactual_sample_weight=counterfactual_sample_weight)

  # Reshape dataset output.
  return dataset.map(_map_fn)


def build_counterfactual_dataset(
    original_dataset,
    sensitive_terms_to_remove: Optional[List[str]] = None,
    custom_counterfactual_function: Optional[Callable[[Any], Any]] = None
    ) -> tf.data.Dataset:
  # pyformat: disable
  """Build Counterfactual dataset from a list sensitive terms or custom function.

  Arguments:
    original_dataset: `tf.data.Dataset` that was used before applying
      Counterfactual. The output should conform to the format used in
      `tf.keras.Model.fit`.
    sensitive_terms_to_remove: List of terms that will be removed or a
      dictionary of terms that will be replaced within the original dataset.
    custom_counterfactual_function: Optional custom function to apply
      to `tf.data.Dataset.map` to build a custom counterfactual dataset.

  This function builds a `tf.data.Dataset` containing examples that are meant to
  only be used when calculating a `counterfactual_loss`. This resulting dataset
  will need to be packed with the original dataset used for the original task of
  the model which can be done by calling `utils.pack_counterfactual_data`.

  Warning: All input datasets should be batched **before** being passed in.

  `original_dataset` must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  This output will be parsed internally in the following way:

  ```
  batch = ...  # Batch from any of the input datasets.
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(batch)
  ```

  Note: the `y` component of input datasets will be ignored completely so it can
  be set to `None` or any other arbitrary value. If `sample_weight` is not
  included, it can be left out entirely.

  Every batch from the returned `tf.data.Dataset` will contain the same batch
  input as before for each dataset, but with a modified counterfactual dataset
  instead of the original text dataset.

  Alternatively to passing a `sensitive_terms_to_remove`, a custom function can
  be created that will be passed to the original to create a counterfactual
  dataset as specficied by the users. For example a users might want to replace
  a target word instead of simply removing the word.

  Returns:
    A `tf.data.Dataset` whose output is a tuple or structure (matching the
      structure of the inputs) of `main: (x, y, w,)
      counterfactual: (x, weight)`.

  Raises:
    ValueError: If both `custom_counterfactual_function` and
      `sensitive_terms_to_remove` are not provided.
    ValueError: If column name is not found within the original dataset.
  """
  # pyformat: enable
  if custom_counterfactual_function is None and not sensitive_terms_to_remove:
    raise ValueError(
        "Either `custom_counterfactual_function` must be provided or "
        "`sensitive_terms_to_remove` and `feature_column` must provided.\n"
        f"Found:\nsensitive_terms_to_remove: {sensitive_terms_to_remove}\n"
        f"custom_counterfactual_function: {custom_counterfactual_function}")

  # Validate original_dataset structures.
  structure_utils.validate_counterfactual_structure(
      original_dataset,
      struct_name="original_dataset",
      element_type=tf.data.Dataset)
  original_dataset = tf.nest.map_structure(
      lambda dataset: dataset, original_dataset)
  counterfactual_dataset = tf.data.Dataset.zip((original_dataset,))

  def _create_counterfactual_dataset(original_batch):
    original_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(original_batch)

    # Mapped function to remove words from the original dataset. If you're
    # interested in modifying the dataset in a different fastion (e.g. replace
    # words) please look into passing a custom funciton via
    # `custom_counterfactual_function`.
    list_regex = "(%s)" % "|".join(sensitive_terms_to_remove)
    counterfactual_x = tf.strings.regex_replace(original_x, list_regex, "")
    return tf.keras.utils.pack_x_y_sample_weight(counterfactual_x)

  def _map_fn(orginal_batch):
    counterfactual_func = (
        custom_counterfactual_function if custom_counterfactual_function
        is not None else _create_counterfactual_dataset)
    return counterfactual_func(orginal_batch)
  return counterfactual_dataset.map(_map_fn)


def unpack_original_x(inputs):
  """Unpacks `original_x` from a `CounterfactualPackedInputs` instance.

  Arguments:
    inputs: CounterfactualPackedInputs to be unpacked, if possible.

  Returns:
    `original_x` if `inputs` is an instance of
      `CounterfactualPackedInputs`, otherwise returns `None`.
  """
  if not isinstance(inputs, CounterfactualPackedInputs):
    return None  # Default to returning None directly.
  return inputs.original_x


def unpack_original_y(inputs):
  """Unpacks `original_y` from a `CounterfactualPackedInputs` instance.

  Arguments:
    inputs: CounterfactualPackedInputs to be unpacked, if possible.

  Returns:
    `original_y` if `inputs` is an instance of
      `CounterfactualPackedInputs`, otherwise returns `None`.
  """
  if not isinstance(inputs, CounterfactualPackedInputs):
    return None  # Default to returning None directly.
  return inputs.original_y


def unpack_original_sample_weight(inputs):
  """Unpacks `original_sample_weight` from a `CounterfactualPackedInputs`.

  Arguments:
    inputs: CounterfactualPackedInputs to be unpacked, if possible.

  Returns:
    `original_sample_weight` if `inputs` is an instance of
      `CounterfactualPackedInputs`, otherwise returns `None`.
  """
  if not isinstance(inputs, CounterfactualPackedInputs):
    return None  # Default to returning None directly.
  return inputs.original_sample_weight


def unpack_counterfactual_x(inputs):
  """Unpacks `counterfactual_x` from a `CounterfactualPackedInputs` instance.

  Arguments:
    inputs: Data to be unpacked, if possible.

  Returns:
    `counterfactual_x` if `inputs` is an instance of
      `CounterfactualPackedInputs`, otherwise returns `None`.
  """
  if not isinstance(inputs, CounterfactualPackedInputs):
    return None  # Default to returning None.
  return inputs.counterfactual_x


def unpack_counterfactual_sample_weight(inputs):
  """Unpacks `counterfactual_sample_weight` from a `CounterfactualPackedInputs`.

  Arguments:
    inputs: CounterfactualPackedInputs to be unpacked, if possible.

  Returns:
    `counterfactual_sample_weight` if `inputs` is an instance of
      `CounterfactualPackedInputs`, otherwise returns `None`.
  """
  if not isinstance(inputs, CounterfactualPackedInputs):
    return None  # Default to returning None.
  return inputs.counterfactual_sample_weight


def unpack_x_y_sample_weight_cfx_cfsample_weight(data):
  """Unpacks user-provided data tuple.

  This is a convenience utility to be used when overriding
  `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
  This utility makes it easy to support data of the form `(x,)`,
  `(x, y)`, `(x, y, sample_weight)`, `(x, y, sample_weight, cfx)`, or
  `(x, y, sample_weight, cfx, cfsample_weight)`.

  Args:
    data: A CounterfactualPackedInputs tuple of the form `(x,)`, `(x, y)`,
      `(x, y, sample_weight)`, `(x, y, sample_weight, cfx)`, or
      `(x, y, sample_weight, cfx, cfsample_weight)`.

  Returns:
    The unpacked tuple, with `None`s for tuple values that are not provided.

  Raises:
    TypeError: If the data is not an instance of CounterfactualPackedInputs.
  """
  if isinstance(data, CounterfactualPackedInputs):
    return (unpack_original_x(data),
            unpack_original_y(data),
            unpack_original_sample_weight(data),
            unpack_counterfactual_x(data),
            unpack_counterfactual_sample_weight(data))
  else:
    error_msg = (
        "Data is expected to be in a CounterfactualPackedInputs format of "
        f"`(x, y, sample_weight, cfx, cfsample_weight)` found: {data}")
    raise TypeError(error_msg)
