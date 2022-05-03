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
        "original_input",
        "counterfactual_data",
    ])):
  """Named tuple containing inputs for the original and counterfactual data.

  `CounterfactualModel` default implementations functions use this class to pack
  and unpack the separate components required for Counterfactual and regular
  training.

  Attributes:
    original_input: Batch of inputs that would originally (i.e. without
      applying Counterfactual) be passed in to a model's `Model.call` method.
      This corresponds to the `x` component described in `tf.keras.Model.fit`.
    counterfactual_data: Counterfactual value based on `original_x` that has
      been modified. Should be the form `(orginal_x, counterfactual_x,
      counterfactual_samle_weight)`.
  """


def pack_counterfactual_data(
    original_input: tf.data.Dataset,
    counterfactual_data: tf.data.Dataset) -> CounterfactualPackedInputs:
  """Packs `counterfactual_data` with the `original_input`.

  Arguments:
    original_input: `tf.data.Dataset` that was used before applying
      Counterfactual. The output should conform to the format used in
      `tf.keras.Model.fit`.
    counterfactual_data: `tf.data.Dataset` or valid Counterfactual structure
      (unnested dict) of `tf.data.Dataset`s containing only examples to be used
      to calculate the `counterfactual_loss`. This dataset is repeated to match
      the number of examples in `original_input`.

  This function should be used to create the dataset that will be passed to
  `counterfactual.keras.CounterfactualModel` during training and, optionally,
  during evaluation.

  Warning: All input datasets should be batched **before** being passed in.

  Each input dataset must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  Every batch from the returned `tf.data.Dataset` will contain one batch from
  each of the input datasets as a `CounterfactualPackedInputs`. Each returned
  batch will be a tuple from the original dataset and counterfactual dataset
  such that `((x, y, sample_weight), (original_x, counterfactual_y,
  counterfactual_sample_weight))` matching the length of `original_input`
  batches where:

  - `original_input`: is a `tf.data.Dataset` that contains:
    - `x`: `x` component taken directly from the `original_input` batch.
    - `y`: `y` component taken directly from the `original_input` batch.
    - `sample_weight`: `sample_weight` component taken directly from the
      `original_input` batch.

  - `counterfactual_data`: is a `tf.data.Dataset` that contains:
    - `original_x`: `x` component taken directly from the
        `original_input` batch.
    - `counterfactual_x`: Batch dataset of data formed from
      `counterfactual_data` (as described in `utils.build_counterfactual_data`).
    - `counterfactual_sample_weight`: Batch of data formed from taken directly
      from the `counterfactual_sample_weight` of `counterfactual_data`.

  Note that the `original_x` should be an `x` value within the original dataset
  that contains an attribute you're looking to apply Counterfactuals to.
  Additionally, `counterfactual_dataset` should only include instances of `x`
  values that have a difference `counterfactual_x`. The shape of the two
  datasets should also be the same, but it is fine (and expected) and there are
  duplicate rows within the `counterfactual_dataset` to match the shape of the
  original dataset.

  `counterfactual_data` will be used in
  `counterfactual.keras.CounterfactualModel` when calculating the
  `counterfactual_loss`.

  Returns:
    A tuple of `tf.data.Dataset` whose output contains ((`x`, `y`,
      `sample_weight`), (`original_x`, `counterfactual_x`,
      `counterfactual_sample_weight`)) matching the output length of
      `original_input`.
  """
  # Validate original_input and counterfactual_data structure.
  structure_utils.validate_counterfactual_structure(
      original_input,
      struct_name="original_input",
      element_type=tf.data.Dataset)
  structure_utils.validate_counterfactual_structure(
      counterfactual_data,
      struct_name="counterfactual_data",
      element_type=tf.data.Dataset)

  tf.nest.assert_same_structure(
      original_input, counterfactual_data)

  dataset = tf.data.Dataset.zip((original_input, counterfactual_data.repeat()))

  def _map_fn(original_batch, counterfactual_batch):
    return CounterfactualPackedInputs(
        original_input=original_batch,
        counterfactual_data=counterfactual_batch)
  return dataset.map(_map_fn)


def build_counterfactual_data(
    original_input: tf.data.Dataset,
    sensitive_terms_to_remove: Optional[List[str]] = None,
    custom_counterfactual_function: Optional[Callable[[Any], Any]] = None
    ) -> tf.data.Dataset:
  # pyformat: disable
  """Build Counterfactual dataset from a list sensitive terms or custom function.

  Arguments:
    original_input: `tf.data.Dataset` that was used before applying
      Counterfactual. The output should conform to the format used in
      `tf.keras.Model.fit`.
    sensitive_terms_to_remove: List of terms that will be removed or a
      dictionary of terms that will be replaced within the original dataset.
    custom_counterfactual_function: Optional custom function to apply
      to `tf.data.Dataset.map` to build a custom counterfactual dataset. Note
      that Needs return dataset must be in the form of `(original_x,
      counterfactual_x, counterfactual_sample_weight)`

  This function builds a `tf.data.Dataset` containing examples that are meant to
  only be used when calculating a `counterfactual_loss`. This resulting dataset
  will need to be packed with the original dataset used for the original task of
  the model which can be done by calling `utils.pack_counterfactual_data`.

  Warning: All input datasets should be batched **before** being passed in.

  `original_input` must output a tuple in the format used in
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
      structure of the inputs) of `main: (x, y, sample_weight)
      counterfactual: (original_x, counterfactual_x,
      counterfactual_sample_weight)`.

  Raises:
    ValueError: If both `custom_counterfactual_function` and
      `sensitive_terms_to_remove` are not provided.
  """
  # pyformat: enable
  if custom_counterfactual_function is None and not sensitive_terms_to_remove:
    raise ValueError(
        "Either `custom_counterfactual_function` must be provided or "
        "`sensitive_terms_to_remove` must provided.\n"
        f"Found:\nsensitive_terms_to_remove: {sensitive_terms_to_remove}\n"
        f"custom_counterfactual_function: {custom_counterfactual_function}")

  # Validate original_input structures.
  structure_utils.validate_counterfactual_structure(
      original_input,
      struct_name="original_input",
      element_type=tf.data.Dataset)
  original_input = tf.nest.map_structure(
      lambda dataset: dataset, original_input)
  counterfactual_data = tf.data.Dataset.zip((original_input,))

  def _create_counterfactual_data(original_batch):
    original_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(original_batch)

    # Mapped function to remove words from the original dataset. If you're
    # interested in modifying the dataset in a different fastion (e.g. replace
    # words) please look into passing a custom funciton via
    # `custom_counterfactual_function`.
    list_regex = "\\b(%s)\\b" % "|".join(sensitive_terms_to_remove)
    counterfactual_x = tf.strings.regex_replace(original_x, list_regex, "")
    return tf.keras.utils.pack_x_y_sample_weight(
        original_x, counterfactual_x, tf.ones_like(original_x, tf.float32))

  def _map_fn(original_batch):
    counterfactual_func = (
        custom_counterfactual_function if custom_counterfactual_function
        is not None else _create_counterfactual_data)
    return counterfactual_func(original_batch)

  def _filter_fn(original_batch):
    original_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(original_batch)
    list_regex = ".*(\\b%s\\b).*" % "|".join(sensitive_terms_to_remove)
    return tf.math.reduce_all(
        tf.strings.regex_full_match(original_x, list_regex))

  if sensitive_terms_to_remove is None:
    return counterfactual_data.map(_map_fn)

  return counterfactual_data.filter(_filter_fn).map(_map_fn)
