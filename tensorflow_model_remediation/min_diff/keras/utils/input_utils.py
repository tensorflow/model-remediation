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

"""Input utils module for MinDiff Keras integration.

This module provides default implementations for packing and unpacking min_diff
data into or from an input dataset.
"""

import collections

import tensorflow as tf


# Convenience class to help with packing and unpacking.
class MinDiffPackedInputs(
    collections.namedtuple("MinDiffPackedInputs",
                           ["original_inputs", "min_diff_data"])):
  """Named tuple containing both `original_inputs` and `min_diff_data`.

  `MinDiffModel` default implementations and `utils.(un)pack_*` functions use
  this class to pack and unpack the separate components required for MinDiff
  and regular training.

  Attributes:
    original_inputs: Batch of inputs that would originally (i.e. without
      applying MinDiff) be passed in to a model's `Model.call` method. This
      corresponds to the `x` component described in `tf.keras.Model.fit`.
    min_diff_data: Batch of supplemental data to be used to calculate the
      `min_diff_loss`.
  """


def pack_min_diff_data(
    original_dataset: tf.data.Dataset, sensitive_group_dataset: tf.data.Dataset,
    nonsensitive_group_dataset: tf.data.Dataset) -> tf.data.Dataset:
  # pyformat: disable
  """Packs `min_diff_data` with the `x` component of the original dataset.

  Arguments:
    original_dataset: `tf.data.Dataset` that was used before applying min
      diff. The output should conform to the format used in
      `tf.keras.Model.fit`.
    sensitive_group_dataset: `tf.data.Dataset` containing only examples that
      belong to the sensitive group. The output should have the same structure
      as that of `original_dataset`.
    nonsensitive_group_dataset: `tf.data.Dataset` containing only examples that
      do not belong to the sensitive group. The output should have the same
      structure as that of `original_dataset.

  This function should be used to create the dataset that will be passed to
  `min_diff.keras.MinDiffModel` during training and, optionally, during
  evaluation.

  Warning: All three input datasets should be batched **before** being passed in.

  Each input dataset must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  This output will be parsed internally in the following way:

  ```
  batch = ...  # Batch from any one of the input datasets.
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(batch)
  ```

  Note: the `y` component of `sensitive_dataset` and `nonsensitive_dataset`
  will be ignored completely so it can be set to `None` or any other arbitrary
  value. If `sample_weight` is not included, it can be left out entirely.

  The `tf.data.Dataset` returned will output a tuple of `(packed_inputs, y,
  sample_weight)` where:

  - `packed_inputs`: is an instance of `utils.MinDiffPackedInputs` containing:

    - `original_inputs`: `x` component from the `original_dataset`.
    - `min_diff_data`: data formed from `sensitive_group_dataset` and
    `nonsensitive_group_dataset` as described below.

  - `y`: is the `y` component taken directly from `original_dataset`.
  - `sample_weight`: is the `sample_weight` component taken directly from
    `original_dataset`.

  `min_diff_data` will be used in `min_diff.keras.MinDiffModel` when calculating
  the `min_diff_loss`. It is a tuple of `(min_diff_x, min_diff_membership,
  min_diff_sample_weight)` where:

  - `min_diff_x`: is formed by concatenating the `x` components of
    `sensitive_dataset` and `nonsensitive_dataset`.
  - `min_diff_membership`: is a tensor of size `[min_diff_batch_size, 1]`
    indicating which dataset each example comes from (`1.0` for
    `sensitive_group_dataset` and `0.0` for `nonsensitive_group_dataset`).
  - `min_diff_sample_weight`: is formed by concatenating the `sample_weight`
    components of `sensitive_dataset`and `nonsensitive_dataset`. If both are
    `None`, then this will be set to `None`. If only one is `None`, it is
    replaced with a `Tensor` of ones of the appropriate shape.

  Returns:
    A `tf.data.Dataset` whose output is a tuple of (`packed_inputs`, `y`,
    `sample_weight`).
  """
  # pyformat: enable

  dataset = tf.data.Dataset.zip(
      (original_dataset, sensitive_group_dataset.repeat(),
       nonsensitive_group_dataset.repeat()))

  # TODO: Should we conserve the length of the tuples returned?
  #                    Right now we always return a tuple of length 3 (with None
  #                    if things are missing).
  def _map_fn(original_batch, sensitive_batch, nonsensitive_batch):
    # Unpack all three batches.
    original_x, original_y, original_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(original_batch))
    sensitive_x, _, sensitive_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(sensitive_batch))
    nonsensitive_x, _, nonsensitive_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(nonsensitive_batch))

    # original_x, sensitive_x and nonsensitive_x all must have the same
    # structure.
    # TODO: Should we assert that Tensor shapes are the same (other
    #                    than number of examples).
    tf.nest.assert_same_structure(original_x, sensitive_x)
    tf.nest.assert_same_structure(sensitive_x, nonsensitive_x)

    # Create min_diff_data.
    # Merge sensitive_x and nonsensitive_x to form min_diff_x.
    flat_sensitive_x = tf.nest.flatten(sensitive_x)
    flat_nonsensitive_x = tf.nest.flatten(nonsensitive_x)
    flat_min_diff_x = [
        tf.concat([t1, t2], axis=0)
        for t1, t2 in zip(flat_sensitive_x, flat_nonsensitive_x)
    ]
    min_diff_x = tf.nest.pack_sequence_as(original_x, flat_min_diff_x)

    # min_diff_membership indicates which dataset each example comes from.
    sensitive_shape = [tf.shape(flat_sensitive_x[0])[0], 1]
    nonsensitive_shape = [tf.shape(flat_nonsensitive_x[0])[0], 1]
    min_diff_membership = tf.concat(
        axis=0,
        values=[
            tf.ones(sensitive_shape, dtype=tf.float32),
            tf.zeros(nonsensitive_shape, dtype=tf.float32)
        ])
    # min_diff_sample_weight is the concatenation of both sample_weights.
    min_diff_sample_weight = None  # Default if both sample_weights are None.
    if (sensitive_sample_weight is not None or
        nonsensitive_sample_weight is not None):
      if sensitive_sample_weight is None:
        sensitive_sample_weight = tf.ones(sensitive_shape, dtype=tf.float32)
      elif nonsensitive_sample_weight is None:
        nonsensitive_sample_weight = tf.ones(
            nonsensitive_shape, dtype=tf.float32)
      min_diff_sample_weight = tf.concat(
          [sensitive_sample_weight, nonsensitive_sample_weight], axis=0)

    # Pack the three components into min_diff_data
    min_diff_data = tf.keras.utils.pack_x_y_sample_weight(
        min_diff_x, min_diff_membership, min_diff_sample_weight)

    # pack min_diff_data with original_x
    return tf.keras.utils.pack_x_y_sample_weight(
        MinDiffPackedInputs(
            original_inputs=original_x, min_diff_data=min_diff_data),
        original_y, original_sample_weight)

  # Reshape dataset output.
  return dataset.map(_map_fn)


def unpack_original_inputs(inputs):
  """Unpacks `original_inputs` from a `utils.MinDiffPackedInputs` instance.

  Arguments:
    inputs: Data to be unpacked, if possible.

  Returns:
    `original_inputs` if `inputs` is an instance of `utils.MinDiffPackedInputs`,
      otherwise `inputs` is returned directly.
  """
  if not isinstance(inputs, MinDiffPackedInputs):
    return inputs  # Default to returning inputs directly.
  return inputs.original_inputs


def unpack_min_diff_data(inputs):
  """Unpacks `min_diff_data` from a `utils.MinDiffPackedInputs` instance.

  Arguments:
    inputs: Data to be unpacked, if possible.

  Returns:
    `min_diff_data` if `inputs` is an instance of `utils.MinDiffPackedInputs`,
      otherwise returns `None`.
  """
  if not isinstance(inputs, MinDiffPackedInputs):
    return None  # Default to returning None.
  return inputs.min_diff_data
