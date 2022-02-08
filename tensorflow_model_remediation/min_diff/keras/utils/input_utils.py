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

"""Input utils module for MinDiff Keras integration.

This module provides default implementations for packing and unpacking min_diff
data into or from an input dataset.
"""

import collections

import tensorflow as tf

from tensorflow_model_remediation.min_diff.keras.utils import structure_utils


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


def pack_min_diff_data(original_dataset: tf.data.Dataset,
                       sensitive_group_dataset=None,
                       nonsensitive_group_dataset=None,
                       min_diff_dataset=None) -> tf.data.Dataset:
  # pyformat: disable
  """Packs `min_diff_data` with the `x` component of the original dataset.

  Arguments:
    original_dataset: `tf.data.Dataset` that was used before applying min
      diff. The output should conform to the format used in
      `tf.keras.Model.fit`.
    sensitive_group_dataset: `tf.data.Dataset` or valid MinDiff structure
      (unnested dict) of `tf.data.Dataset`s containing only examples that
      belong to the sensitive group.

      This must be passed in if `nonsensitive_group_dataset` is passed in.
      Furthermore, the `x` component for every batch should have the same
      structure as that of the `original_dataset` batches' `x` components.
    nonsensitive_group_dataset: `tf.data.Dataset` or valid MinDiff structure
      (unnested dict) of `tf.data.Dataset`s containing only examples that do
      **not** belong to the sensitive group.

      This must be passed in if `sensitive_group_dataset` is passed in.
      Furthermore, the `x` component for every batch should have the same
      structure as that of the `original_dataset` batches' `x` components.
    min_diff_dataset: `tf.data.Dataset` or valid MinDiff structure (unnested
      dict) of `tf.data.Dataset`s containing only examples to be used to
      calculate the `min_diff_loss`.

      This should only be set if neither `sensitive_group_dataset` or
      `nonsensitive_group_dataset` is passed in.
      Furthermore, the `x` component for every batch should have the same
      structure as that of the `original_dataset` batches' `x` components.

  This function should be used to create the dataset that will be passed to
  `min_diff.keras.MinDiffModel` during training and, optionally, during
  evaluation.

  The inputs should either have both `sensitive_group_dataset` and
  `nonsensitive_group_dataset` passed in and `min_diff_dataset` left unset or
  vice versa. In the case of the former, `min_diff_data` will be built using
  `utils.build_min_diff_dataset`.

  Warning: All input datasets should be batched **before** being passed in.

  Each input dataset must output a tuple in the format used in
  `tf.keras.Model.fit`. Specifically the output must be a tuple of
  length 1, 2 or 3 in the form `(x, y, sample_weight)`.

  This output will be parsed internally in the following way:

  ```
  batch = ...  # Batch from any one of the input datasets.
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(batch)
  ```

  Every batch from the returned `tf.data.Dataset` will contain one batch from
  each of the input datasets. Each returned batch will be a tuple of
  `(packed_inputs, original_y, original_sample_weight)` matching the length of
  `original_dataset` batches where:

  - `packed_inputs`: is an instance of `utils.MinDiffPackedInputs` containing:

    - `original_inputs`: `x` component taken directly from the
        `original_dataset` batch.
    - `min_diff_data`: batch of data formed from `sensitive_group_dataset` and
      `nonsensitive_group_dataset` (as described in
      `utils.build_min_diff_dataset`) or taken directly from `min_diff_dataset`.

  - `original_y`: is the `y` component taken directly from the
    `original_dataset` batch.
  - `original_sample_weight`: is the `sample_weight` component taken directly
    from the `original_dataset` batch.

  `min_diff_data` will be used in `min_diff.keras.MinDiffModel` when calculating
  the `min_diff_loss`. It is a tuple or structure (matching the structure of the
  inputs) of `(min_diff_x, min_diff_membership, min_diff_sample_weight)`.

  Caution: If you are passing in `min_diff_dataset` make sure that each
  `min_diff_data` batch contains about the same number of sensitive and
  nonsensitive examples as indicated by `min_diff_membership` (when passing in
  `sensitive_group_dataset` and `nonsensitive_group_dataset` this is determined
  by their batch sizes).

  Returns:
    A `tf.data.Dataset` whose output is a tuple of (`packed_inputs`,
      `original_y`, `original_sample_weight`) matching the output length
      of `original_dataset`.
  """
  # pyformat: enable
  # Either sensitive_group_dataset and nonsensitive_group_dataset are both set
  # and min_diff_dataset is not or vice versa.
  min_diff_dataset_present = min_diff_dataset is not None
  sensitive_dataset_present = sensitive_group_dataset is not None
  nonsensitive_dataset_present = nonsensitive_group_dataset is not None
  # Case where min_diff_dataset is set and the others are not.
  set_to_use_min_diff_dataset = (
      min_diff_dataset_present and
      not (sensitive_dataset_present or nonsensitive_dataset_present))
  # Case where sensitive_group_dataset and nonsensitive_group_dataset are both
  # set and min_diff_dataset is not.
  set_to_construct_min_diff_dataset = ((sensitive_dataset_present and
                                        nonsensitive_dataset_present) and
                                       not min_diff_dataset_present)
  if not (set_to_use_min_diff_dataset or set_to_construct_min_diff_dataset):
    raise ValueError(
        "Invalid arguments: You must either pass in only the `min_diff_dataset`"
        " (and leave `sensitive_group_dataset` and `nonsensitive_group_dataset`"
        " as None) or set both `sensitive_group_dataset` and "
        "`nonsensitive_group_dataset` (and leave `min_diff_dataset` as None), "
        "given: \n"
        "\n`sensitive_group_dataset`: {}"
        "\n`nonsensitive_group_dataset`: {}"
        "\n`min_diff_dataset`: {}".format(sensitive_group_dataset,
                                          nonsensitive_group_dataset,
                                          min_diff_dataset))

  # First construct the min_diff_dataset if need be.
  if set_to_construct_min_diff_dataset:
    min_diff_dataset = build_min_diff_dataset(sensitive_group_dataset,
                                              nonsensitive_group_dataset)
  else:
    # validate min_diff_dataset since it was passed in.
    structure_utils.validate_min_diff_structure(
        min_diff_dataset,
        struct_name="min_diff_dataset",
        element_type=tf.data.Dataset)

  dataset = tf.data.Dataset.zip((original_dataset, min_diff_dataset))

  def _map_fn(original_batch, min_diff_batch):
    # Unpack original batch.
    original_x, original_y, original_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(original_batch))

    # Assert that all min_diff_xs have the same structure as original_x.
    # TODO: Should we assert that Tensor shapes are the same (other
    #                    than number of examples).

    min_diff_xs = [
        tf.keras.utils.unpack_x_y_sample_weight(batch)[0]  # First element is x.
        for batch in structure_utils._flatten_min_diff_structure(min_diff_batch)
    ]
    for min_diff_x in min_diff_xs:
      try:
        tf.nest.assert_same_structure(original_x, min_diff_x)
      except Exception as e:
        raise type(e)(
            "The x component structure of (one of) the `min_diff_dataset`(s) "
            "does not match that of the original x structure (original shown "
            "first): {}".format(e))

    # pack min_diff_batch with original_x
    return _pack_as_original(
        original_batch,
        MinDiffPackedInputs(
            original_inputs=original_x, min_diff_data=min_diff_batch),
        original_y, original_sample_weight)

  # Reshape dataset output.
  return dataset.map(_map_fn)


def _pack_as_original(original_batch, x, y, w):
  """Packs x, y, w while conserving the shape of the original batch."""
  if not isinstance(original_batch, tuple):
    return x
  length = len(original_batch)
  return (x, y, w)[:length]


def _tensor_concat(t1, t2):
  """Concatenates (sparse or dense) tensors."""
  if isinstance(t1, tf.SparseTensor):
    # Ensure SparseTensors have the same non-batch dim before concatenating.
    max_shape = tf.math.maximum(t1.dense_shape[1], t2.dense_shape[1])
    t1 = tf.sparse.reset_shape(t1, [t1.dense_shape[0], max_shape])
    t2 = tf.sparse.reset_shape(t2, [t2.dense_shape[0], max_shape])
    return tf.sparse.concat(axis=0, sp_inputs=[t1, t2])
  else:
    return tf.concat([t1, t2], axis=0)


def build_min_diff_dataset(sensitive_group_dataset,
                           nonsensitive_group_dataset) -> tf.data.Dataset:
  # pyformat: disable
  """Build MinDiff dataset from sensitive and nonsensitive datasets.

  Arguments:
    sensitive_group_dataset: `tf.data.Dataset` or valid MinDiff structure
      (unnested dict) of `tf.data.Dataset`s containing only examples that
      belong to the sensitive group.
    nonsensitive_group_dataset: `tf.data.Dataset` or valid MinDiff structure
      (unnested dict) of `tf.data.Dataset`s containing only examples that do
      **not** belong to the sensitive group.

  This function builds a `tf.data.Dataset` containing examples that are meant to
  only be used when calculating a `min_diff_loss`. This resulting dataset will
  need to be packed with the original dataset used for the original task of the
  model which can be done by calling `utils.pack_min_diff_data`.

  Warning: All input datasets should be batched **before** being passed in.

  Each input dataset must output a tuple in the format used in
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

  Every batch from the returned `tf.data.Dataset` will contain one batch from
  each of the input datasets. Each returned batch will be a tuple or structure
  (matching the structure of the inputs) of `(min_diff_x, min_diff_membership,
  min_diff_sample_weight)` where, for each pair of input datasets:

  - `min_diff_x`: is formed by concatenating the `x` components of the paired
    datasets. The structure of these must match. If they don't the dataset will
    raise an error at the first batch.
  - `min_diff_membership`: is a tensor of size `[min_diff_batch_size, 1]`
    indicating which dataset each example comes from (`1.0` for
    `sensitive_group_dataset` and `0.0` for `nonsensitive_group_dataset`).
  - `min_diff_sample_weight`: is formed by concatenating the `sample_weight`
    components of the paired datasets. If both are `None`, then this will be set
    to `None`. If only one is `None`, it is replaced with a `Tensor` of ones of
    the appropriate shape.

  Returns:
    A `tf.data.Dataset` whose output is a tuple or structure (matching the
      structure of the inputs) of `(min_diff_x, min_diff_membership,
      min_diff_sample_weight)`.

  Raises:
    ValueError: If either `sensitive_group_dataset` or
      `nonsensitive_group_dataset` is not a valid MinDiff structure (unnested
      dict).
    ValueError: If `sensitive_group_dataset` and `nonsensitive_group_dataset` do
      not have the same structure.
  """
  # pyformat: enable
  # validate structures.
  structure_utils.validate_min_diff_structure(
      sensitive_group_dataset,
      struct_name="sensitive_group_dataset",
      element_type=tf.data.Dataset)
  structure_utils.validate_min_diff_structure(
      nonsensitive_group_dataset,
      struct_name="nonsensitive_group_dataset",
      element_type=tf.data.Dataset)
  try:

    structure_utils._assert_same_min_diff_structure(sensitive_group_dataset,
                                                    nonsensitive_group_dataset)
  except Exception as e:
    raise type(e)("`sensitive_group_dataset` and `nonsensitive_group_dataset` "
                  "do not have the same structure:\n{}".format(e))

  sensitive_group_dataset = tf.nest.map_structure(
      lambda dataset: dataset, sensitive_group_dataset)
  nonsensitive_group_dataset = tf.nest.map_structure(
      lambda dataset: dataset, nonsensitive_group_dataset)

  dataset = tf.data.Dataset.zip(
      (sensitive_group_dataset, nonsensitive_group_dataset))

  def _build_single_batch(single_sensitive_batch, single_nonsensitive_batch):
    # Unpack both batches.
    sensitive_x, _, sensitive_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(single_sensitive_batch))
    nonsensitive_x, _, nonsensitive_sample_weight = (
        tf.keras.utils.unpack_x_y_sample_weight(single_nonsensitive_batch))

    # sensitive_x and nonsensitive_x must have the same structure.
    try:
      tf.nest.assert_same_structure(sensitive_x, nonsensitive_x)
    except Exception as e:
      raise type(e)("The x component structure of (one of) the "
                    "`sensitive_group_dataset`(s) does not match that of the "
                    "(corresponding) `nonsensitive_group_dataset` x structure "
                    "(sensitive shown first): {}".format(e))

    # Create min_diff_data.
    # Merge sensitive_x and nonsensitive_x to form min_diff_x.
    flat_sensitive_x = tf.nest.flatten(sensitive_x)
    flat_nonsensitive_x = tf.nest.flatten(nonsensitive_x)
    flat_min_diff_x = [
        _tensor_concat(t1, t2)
        for t1, t2 in zip(flat_sensitive_x, flat_nonsensitive_x)
    ]
    min_diff_x = tf.nest.pack_sequence_as(sensitive_x, flat_min_diff_x)

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

    # Pack the three components and return them
    return tf.keras.utils.pack_x_y_sample_weight(min_diff_x,
                                                 min_diff_membership,
                                                 min_diff_sample_weight)

  def _map_fn(sensitive_batch, nonsensitive_batch):

    flat_sensitive_batch = structure_utils._flatten_min_diff_structure(
        sensitive_batch)
    flat_nonsensitive_batch = structure_utils._flatten_min_diff_structure(
        nonsensitive_batch)

    flat_min_diff_data = [
        _build_single_batch(single_sensitive_batch, single_nonsensitive_batch)
        for single_sensitive_batch, single_nonsensitive_batch in zip(
            flat_sensitive_batch, flat_nonsensitive_batch)
    ]

    return structure_utils._pack_min_diff_sequence_as(sensitive_batch,
                                                      flat_min_diff_data)

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
