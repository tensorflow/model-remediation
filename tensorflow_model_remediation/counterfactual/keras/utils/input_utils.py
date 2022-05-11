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
      counterfactual_sample_weight)`.
  """


def pack_counterfactual_data(
    original_input: tf.data.Dataset,
    counterfactual_data: tf.data.Dataset) -> tf.data.Dataset:
  """Packs `counterfactual_data` with the `original_input`.

  Arguments:
    original_input: An instance of `tf.data.Dataset` that was used for training
      the original model. The output should conform to the format used in
      `tf.keras.Model.fit`.
    counterfactual_data: An instance of `tf.data.Dataset` containing only
      examples that will be used to calculate the `counterfactual_loss`. This
      dataset is repeated to match the number of examples in `original_input`.

  This function should be used to create an instance of
  `CounterfactualPackedInputs` that will be passed to
  `counterfactual.keras.CounterfactualModel` during training and, optionally,
  during evaluation.

  Each `original_input` must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  Every batch from the returned `tf.data.Dataset` will contain one batch from
  each of the input datasets as a `CounterfactualPackedInputs`. Each returned
  batch will be a tuple from the original dataset and counterfactual dataset
  of format `((x, y, sample_weight), (original_x, counterfactual_x,
  counterfactual_sample_weight))` matching the length of `original_input`
  batches where:

  - `original_input`: is a `tf.data.Dataset` that contains:
    - `x`: The `x` component taken directly from the `original_input` batch.
    - `y`: The `y` component taken directly from the `original_input` batch.
    - `sample_weight`: The `sample_weight` component taken directly from the
      `original_input` batch.

  - `counterfactual_data`: is a `tf.data.Dataset` that contains:
    - `original_x`: The `x` component taken directly from the
        `original_input` batch.
    - `counterfactual_x`: The counterfactual value for `original_x` (as
         described in `build_counterfactual_data`).
    - `counterfactual_sample_weight`: Batch of data formed from taken directly
         from the `counterfactual_sample_weight` of `counterfactual_data`.

  Note: the `original_x` does not need to be an `x` value within the
  original dataset. Additionally, `counterfactual_dataset` should only include
  instances of `x` values that have a difference `counterfactual_x`. It is fine
  (and expected) within `counterfactual_data` to include duplicate rows to
  match the shape of the original dataset.

  The return of `counterfactual_data` will be an instance of
  `CounterfactualPackedInputs` that can be used in
  `counterfactual.keras.CounterfactualModel` when calculating the
  `counterfactual_loss`.

  Returns:
    A `tf,data,Dataset` of `CounterfactualPackedInputs`. Each
    `CounterfactualPackedInputs` represents a
    `(original_inputs, counterfactual_data)` pair where `original_inputs is
    a `(x, y, sample_weight)` tuple, and `counterfactual_data` is a
    `(original_x, counterfactual_x, counterfactual_sample_weight)` tuple.
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
    sensitive_terms_to_remove: List of terms that will be removed and filtered
      from within the `original_input`.
    custom_counterfactual_function: Optional custom function to apply
      to `tf.data.Dataset.map` to build a custom counterfactual dataset. Note
      that it needs to  return a dataset in the form of `(original_x,
      counterfactual_x, counterfactual_sample_weight)` and should only include
      values that have been modified. Use `sensitive_terms_to_remove` to filter
      values that have modifying terms included.

  This function builds a `tf.data.Dataset` containing only examples that will be
  used when calculating `counterfactual_loss`. This resulting dataset
  will need to be packed with the `x` value in `original_input`, a modified
  version of `x` that will act as `counterfactual_x`, and a
  `counterfactual_sample_weight` that defaults to 1.0. The resulting dataset can
  be passed to `pack_counterfactual_data` to create an instance of
  `CounterfactualPackedInputs` for use within
  `counterfactual.keras.CounterfactualModel`.

  `original_input` must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`. This output will be
  parsed internally in the following way:

  ```
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(batch)
  ```

  Alternatively to passing a `sensitive_terms_to_remove`, you can create
  a custom function that you can pass to the original to create a
  counterfactual dataset as specficied by the users. For example, you
  might want to replace a target word instead of simply removing the word.
  The returned `tf.data.Dataset` will need to have the unchanged `x`
  values removed. Passing `sensitive_terms_to_remove` in this case
  acts like a filter to only include terms that have been modified.

  A minimal example is given below:

  >>> simple_dataset_x = tf.constant(["Bad word", "Good word"])

  >>> simple_dataset = tf.data.Dataset.from_tensor_slices((simple_dataset_x))

  >>> counterfactual_data = counterfactual.keras.utils.build_counterfactual_data(
  ...   original_input=simple_dataset, sensitive_terms_to_remove=['Bad'])

  >>> for original_value, counterfactual_value, _ in counterfactual_data.take(1):
  ...   print("original: ", original_value)
  ...   print("counterfactual: ", counterfactual_value)
  ...   print("counterfactual_sample_weight: ", cf_weight)
  original:  tf.Tensor(b'Bad word', shape=(), dtype=string)
  counterfactual:  tf.Tensor(b' word', shape=(), dtype=string)
  counterfactual_sample_weight:  tf.Tensor(1.0, shape=(), dtype=float32)

  Returns:
    A `tf.data.Dataset` whose output is a tuple matching `(original_x,
      counterfactual_x, counterfactual_sample_weight)`.

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
