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

"""Tests for input_utils functions."""

import copy
import tensorflow as tf

from tensorflow_model_remediation.min_diff.keras.utils import input_utils


def _get_batch(tensors, batch_size, batch_num):
  if isinstance(tensors, dict):
    return {k: _get_batch(v, batch_size, batch_num) for k, v in tensors.items()}

  if isinstance(tensors, tf.SparseTensor):
    total_examples = tensors.dense_shape[0]
  else:
    total_examples = len(tensors)
  start_ind = (batch_size * batch_num) % total_examples
  end_ind = start_ind + batch_size
  # Double tensor to enable repeating inputs.
  if isinstance(tensors, tf.SparseTensor):
    return tf.sparse.slice(tensors, [start_ind, 0], [batch_size, 2])
  else:
    return tensors[start_ind:end_ind]


def _get_min_diff_batch(sensitive_tensors, nonsensitive_tensors,
                        sensitive_batch_size, nonsensitive_batch_size,
                        batch_num):
  sensitive_batch = _get_batch(sensitive_tensors, sensitive_batch_size,
                               batch_num)
  nonsensitive_batch = _get_batch(nonsensitive_tensors, nonsensitive_batch_size,
                                  batch_num)
  if isinstance(sensitive_batch, dict):
    return {
        key: input_utils._tensor_concat(sensitive_batch[key],
                                        nonsensitive_batch[key])
        for key in sensitive_batch.keys()
    }
  return input_utils._tensor_concat(sensitive_batch, nonsensitive_batch)


def _get_min_diff_membership_batch(sensitive_batch_size,
                                   nonsensitive_batch_size):
  return tf.concat(
      axis=0,
      values=[
          tf.ones([sensitive_batch_size, 1], tf.float32),
          tf.zeros([nonsensitive_batch_size, 1], tf.float32)
      ])


class MinDiffInputUtilsTestCase(tf.test.TestCase):

  def assertAllClose(self, a, b):
    """Recursive comparison that handles SparseTensors."""
    if isinstance(a, dict):
      for key, a_value in a.items():
        b_value = b.get(key)
        self.assertIsNotNone(b_value)
        self.assertAllClose(a_value, b_value)
    elif isinstance(a, tf.SparseTensor):
      self.assertAllEqual(a.indices, b.indices)
      super().assertAllClose(a.values, b.values)
      self.assertAllEqual(a.dense_shape, b.dense_shape)
    else:
      super().assertAllClose(a, b)

  def setUp(self):
    super().setUp()

    def to_sparse(tensor):
      """Helper to create a SparseTensor from a dense Tensor."""
      values = tf.reshape(tensor, [-1])
      indices = tf.where(tensor)
      shape = tf.shape(tensor, out_type=tf.int64)
      return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    # Original inputs with 25 examples. Values go from 100.0 to 299.0.
    self.original_x = {
        "f1": tf.reshape(tf.range(100.0, 175.0), [25, 3]),
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2]))
    }
    self.original_y = tf.reshape(tf.range(225.0, 275.0), [25, 2])
    self.original_w = tf.reshape(tf.range(275.0, 300.0), [25, 1])

    # Sensitive inputs with 15 examples. Values go from 300.0 to 399.0.
    self.sensitive_x = {
        "f1": tf.reshape(tf.range(300.0, 345.0), [15, 3]),
        "f2": tf.reshape(tf.range(345.0, 375.0), [15, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(345.0, 375.0), [15, 2]))
    }
    self.sensitive_w = tf.reshape(tf.range(375.0, 390.0), [15, 1])

    # Nonsensitive inputs with 10 examples. Values go from 400.0 to 499.0.
    self.nonsensitive_x = {
        "f1": tf.reshape(tf.range(400.0, 430.0), [10, 3]),
        "f2": tf.reshape(tf.range(430.0, 450.0), [10, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(430.0, 450.0), [10, 2]))
    }
    self.nonsensitive_w = tf.reshape(tf.range(450.0, 460.0), [10, 1])


class BuildMinDiffDatasetTest(MinDiffInputUtilsTestCase):

  def testBuildFromSingleDatasets(self):
    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 1
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)

    for batch_ind, min_diff_batch in enumerate(dataset):
      # Assert min_diff batch properly formed.
      min_diff_x, min_diff_membership, min_diff_w = min_diff_batch

      self.assertAllClose(
          min_diff_x,
          _get_min_diff_batch(self.sensitive_x, self.nonsensitive_x,
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))
      self.assertAllClose(
          min_diff_membership,
          _get_min_diff_membership_batch(sensitive_batch_size,
                                         nonsensitive_batch_size))
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))

  def testBuildFromDictsOfDatasets(self):
    sensitive_batch_sizes = [3, 5]
    sensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.sensitive_x, None, self.sensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], sensitive_batch_sizes)
    }

    nonsensitive_batch_sizes = [1, 2]
    nonsensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.nonsensitive_x, None, self.nonsensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], nonsensitive_batch_sizes)
    }

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)

    for batch_ind, min_diff_batches in enumerate(dataset):
      min_diff_keys = sorted(min_diff_batches.keys())
      # Assert min_diff_batches has the right structure (i.e. set of keys).
      self.assertAllEqual(min_diff_keys, ["k1", "k2"])

      min_diff_batches = [min_diff_batches[key] for key in min_diff_keys]
      for sensitive_batch_size, nonsensitive_batch_size, min_diff_batch in zip(
          sensitive_batch_sizes, nonsensitive_batch_sizes, min_diff_batches):
        # Assert min_diff batch properly formed.
        min_diff_x, min_diff_membership, min_diff_w = min_diff_batch

        self.assertAllClose(
            min_diff_x,
            _get_min_diff_batch(self.sensitive_x, self.nonsensitive_x,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))
        self.assertAllClose(
            min_diff_membership,
            _get_min_diff_membership_batch(sensitive_batch_size,
                                           nonsensitive_batch_size))
        self.assertAllClose(
            min_diff_w,
            _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))

  def testWithVariableSizeSparseTensors(self):
    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, self.sensitive_w, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_x = copy.copy(self.nonsensitive_x)
    # Modify so that f2_sparse has a different dense shape in non_sensitive
    # than in sensitive.
    nonsensitive_x["f2_sparse"] = tf.sparse.reset_shape(
        nonsensitive_x["f2_sparse"], [10, 5])
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)
    for _, min_diff_batch in enumerate(dataset.take(10)):
      min_diff_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(min_diff_batch)
      self.assertEqual(min_diff_x["f2_sparse"].dense_shape[1], 5)

  def testWithOnlySensitiveWeightsNone(self):
    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)

    for batch_ind, min_diff_batch in enumerate(dataset):
      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(min_diff_batch)
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(
              tf.fill([sensitive_batch_size, 1], 1.0), self.nonsensitive_w,
              sensitive_batch_size, nonsensitive_batch_size, batch_ind))

  def testWithOnlyNonsensitiveWeightsNone(self):
    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)

    for batch_ind, min_diff_batch in enumerate(dataset):
      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(min_diff_batch)
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(self.sensitive_w,
                              tf.fill([nonsensitive_batch_size, 1],
                                      1.0), sensitive_batch_size,
                              nonsensitive_batch_size, batch_ind))

  def testWithBothMinDiffWeightsNone(self):
    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.build_min_diff_dataset(sensitive_dataset,
                                                 nonsensitive_dataset)

    # The resulting dataset will repeat infinitely so we only take the first 10
    # batches which corresponds to 2 full epochs of the nonsensitive dataset.
    for min_diff_batch in dataset.take(10):
      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(min_diff_batch)
      self.assertIsNone(min_diff_w)

  def testDifferentWeightsShapeRaisesError(self):
    sensitive_batch_size = 3
    # Create weights with different shape.
    sensitive_w = self.sensitive_w[:, tf.newaxis, :]
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    with self.assertRaisesRegex(ValueError, "must be rank.*but is rank"):
      _ = input_utils.build_min_diff_dataset(sensitive_dataset,
                                             nonsensitive_dataset)

  def testInvalidStructureRaisesError(self):
    # Input dataset, content doesn't matter.
    inputs = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(5)

    nested_inputs = {"a": inputs, "b": inputs}
    bad_nested_inputs = {"a": inputs, "b": [inputs]}

    # No errors raised for valid nested structures.
    _ = input_utils.build_min_diff_dataset(
        sensitive_group_dataset=nested_inputs,
        nonsensitive_group_dataset=nested_inputs)

    # Assert raises error for invalid sensitive_group_dataset structure.
    with self.assertRaisesRegex(
        ValueError, "sensitive_group_dataset.*unnested"
        ".*only elements of type.*Dataset.*Given"):
      _ = input_utils.build_min_diff_dataset(bad_nested_inputs, nested_inputs)

    # Assert raises error for invalid nonsensitive_group_dataset structure.
    with self.assertRaisesRegex(
        ValueError, "nonsensitive_group_dataset.*unnested.*only elements of "
        "type.*Dataset.*Given"):
      _ = input_utils.build_min_diff_dataset(nested_inputs, bad_nested_inputs)

    # Assert raises error for different sensitive and nonsensitive structures.
    different_nested_inputs = {"a": inputs, "c": inputs}
    with self.assertRaisesRegex(
        ValueError, "sensitive_group_dataset.*"
        "nonsensitive_group_dataset.*do "
        "not have the same structure(.|\n)*don't have the same set of keys"):
      _ = input_utils.build_min_diff_dataset(nested_inputs,
                                             different_nested_inputs)


class PackMinDiffDataTest(MinDiffInputUtilsTestCase):

  def testPackSingleDatasetsWithXAsTensor(self):

    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], self.original_y,
         self.original_w)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x["f1"], None,
         self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x["f1"], None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch_ind, (packed_inputs, y, w) in enumerate(dataset):
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Assert original batch is conserved
      self.assertAllClose(
          packed_inputs.original_inputs,
          _get_batch(self.original_x["f1"], original_batch_size, batch_ind))
      self.assertAllClose(
          y, _get_batch(self.original_y, original_batch_size, batch_ind))
      self.assertAllClose(
          w, _get_batch(self.original_w, original_batch_size, batch_ind))

      # Assert min_diff batch properly formed.
      min_diff_x, min_diff_membership, min_diff_w = packed_inputs.min_diff_data

      self.assertAllClose(
          min_diff_x,
          _get_min_diff_batch(self.sensitive_x["f1"], self.nonsensitive_x["f1"],
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))
      self.assertAllClose(
          min_diff_membership,
          _get_min_diff_membership_batch(sensitive_batch_size,
                                         nonsensitive_batch_size))
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))

  def testPackSingleDatasetsWithXAsDict(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y,
         self.original_w)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 1
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch_ind, (packed_inputs, y, w) in enumerate(dataset):
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Assert original batch is conserved
      self.assertAllClose(
          packed_inputs.original_inputs,
          _get_batch(self.original_x, original_batch_size, batch_ind))
      self.assertAllClose(
          y, _get_batch(self.original_y, original_batch_size, batch_ind))
      self.assertAllClose(
          w, _get_batch(self.original_w, original_batch_size, batch_ind))

      # Assert min_diff batch properly formed.
      min_diff_x, min_diff_membership, min_diff_w = packed_inputs.min_diff_data

      self.assertAllClose(
          min_diff_x,
          _get_min_diff_batch(self.sensitive_x, self.nonsensitive_x,
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))
      self.assertAllClose(
          min_diff_membership,
          _get_min_diff_membership_batch(sensitive_batch_size,
                                         nonsensitive_batch_size))
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                              sensitive_batch_size, nonsensitive_batch_size,
                              batch_ind))

  def testPackSeparateDictsOfDatasets(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y,
         self.original_w)).batch(original_batch_size)

    sensitive_batch_sizes = [3, 5]
    sensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.sensitive_x, None, self.sensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], sensitive_batch_sizes)
    }

    nonsensitive_batch_sizes = [1, 2]
    nonsensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.nonsensitive_x, None, self.nonsensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], nonsensitive_batch_sizes)
    }

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch_ind, (packed_inputs, y, w) in enumerate(dataset):
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Assert original batch is conserved
      self.assertAllClose(
          packed_inputs.original_inputs,
          _get_batch(self.original_x, original_batch_size, batch_ind))
      self.assertAllClose(
          y, _get_batch(self.original_y, original_batch_size, batch_ind))
      self.assertAllClose(
          w, _get_batch(self.original_w, original_batch_size, batch_ind))

      min_diff_keys = sorted(packed_inputs.min_diff_data.keys())
      # Assert min_diff_batches has the right structure (i.e. set of keys).
      self.assertAllEqual(min_diff_keys, ["k1", "k2"])

      min_diff_batches = [
          packed_inputs.min_diff_data[key] for key in min_diff_keys
      ]
      for sensitive_batch_size, nonsensitive_batch_size, min_diff_batch in zip(
          sensitive_batch_sizes, nonsensitive_batch_sizes, min_diff_batches):

        # Assert min_diff batch properly formed.
        min_diff_x, min_diff_membership, min_diff_w = min_diff_batch

        self.assertAllClose(
            min_diff_x,
            _get_min_diff_batch(self.sensitive_x, self.nonsensitive_x,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))
        self.assertAllClose(
            min_diff_membership,
            _get_min_diff_membership_batch(sensitive_batch_size,
                                           nonsensitive_batch_size))
        self.assertAllClose(
            min_diff_w,
            _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))

  def testPackDictsOfDatasets(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y,
         self.original_w)).batch(original_batch_size)

    sensitive_batch_sizes = [3, 5]
    sensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.sensitive_x, None, self.sensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], sensitive_batch_sizes)
    }

    nonsensitive_batch_sizes = [1, 2]
    nonsensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.nonsensitive_x, None, self.nonsensitive_w)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], nonsensitive_batch_sizes)
    }

    dataset = input_utils.pack_min_diff_data(
        original_dataset,
        min_diff_dataset=input_utils.build_min_diff_dataset(
            sensitive_dataset, nonsensitive_dataset))

    for batch_ind, (packed_inputs, y, w) in enumerate(dataset):
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Assert original batch is conserved
      self.assertAllClose(
          packed_inputs.original_inputs,
          _get_batch(self.original_x, original_batch_size, batch_ind))
      self.assertAllClose(
          y, _get_batch(self.original_y, original_batch_size, batch_ind))
      self.assertAllClose(
          w, _get_batch(self.original_w, original_batch_size, batch_ind))

      min_diff_keys = sorted(packed_inputs.min_diff_data.keys())
      # Assert min_diff_batches has the right structure (i.e. set of keys).
      self.assertAllEqual(min_diff_keys, ["k1", "k2"])

      min_diff_batches = [
          packed_inputs.min_diff_data[key] for key in min_diff_keys
      ]
      for sensitive_batch_size, nonsensitive_batch_size, min_diff_batch in zip(
          sensitive_batch_sizes, nonsensitive_batch_sizes, min_diff_batches):

        # Assert min_diff batch properly formed.
        min_diff_x, min_diff_membership, min_diff_w = min_diff_batch

        self.assertAllClose(
            min_diff_x,
            _get_min_diff_batch(self.sensitive_x, self.nonsensitive_x,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))
        self.assertAllClose(
            min_diff_membership,
            _get_min_diff_membership_batch(sensitive_batch_size,
                                           nonsensitive_batch_size))
        self.assertAllClose(
            min_diff_w,
            _get_min_diff_batch(self.sensitive_w, self.nonsensitive_w,
                                sensitive_batch_size, nonsensitive_batch_size,
                                batch_ind))

  def testWithoutOriginalWeights(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 1
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch in dataset:
      # Only validate original batch weights (other tests cover others).
      # Should be of length 2.
      self.assertLen(batch, 2)
      _, _, w = tf.keras.utils.unpack_x_y_sample_weight(batch)

      self.assertIsNone(w)

  def testWithoutOriginalLabels(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        self.original_x).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 1
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch in dataset:
      # Only validate original batch weights (other tests cover others).
      # Should not be a tuple.
      self.assertIsInstance(batch, input_utils.MinDiffPackedInputs)
      _, _, w = tf.keras.utils.unpack_x_y_sample_weight(batch)

      self.assertIsNone(w)

  def testWithOriginalWeightsNone(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 1
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch in dataset:
      # Only validate original batch weights (other tests cover others).
      # Should be of length 3 despite sample_weight being None.
      self.assertLen(batch, 3)
      _, _, w = tf.keras.utils.unpack_x_y_sample_weight(batch)

      self.assertIsNone(w)

  def testBadDatasetCombinationRaisesError(self):
    # Input dataset, content doesn't matter.
    inputs = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(5)

    # No errors raised for correct combination of elements.
    _ = input_utils.pack_min_diff_data(inputs, min_diff_dataset=inputs)
    _ = input_utils.pack_min_diff_data(
        inputs,
        sensitive_group_dataset=inputs,
        nonsensitive_group_dataset=inputs)

    # Assert raised if no datasets provided.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "None.*\n.*nonsensitive_group_dataset.*None.*\n"
        ".*min_diff_dataset.*None"):
      _ = input_utils.pack_min_diff_data(inputs)

    # Assert raised if all datasets provided.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "Dataset.*\n.*nonsensitive_group_dataset.*Dataset.*\n"
        ".*min_diff_dataset.*Dataset"):
      _ = input_utils.pack_min_diff_data(inputs, inputs, inputs, inputs)

    # Assert raised if only sensitive_group_dataset provided.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "Dataset.*\n.*nonsensitive_group_dataset.*None.*\n"
        ".*min_diff_dataset.*None"):
      _ = input_utils.pack_min_diff_data(inputs, sensitive_group_dataset=inputs)

    # Assert raised if only nonsensitive_group_dataset provided.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "None.*\n.*nonsensitive_group_dataset.*Dataset.*\n"
        ".*min_diff_dataset.*None"):
      _ = input_utils.pack_min_diff_data(
          inputs, nonsensitive_group_dataset=inputs)

    # Assert raised if only nonsensitive_group_dataset is missing.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "Dataset.*\n.*nonsensitive_group_dataset.*None.*\n"
        ".*min_diff_dataset.*Dataset"):
      _ = input_utils.pack_min_diff_data(
          inputs, sensitive_group_dataset=inputs, min_diff_dataset=inputs)

    # Assert raised if only sensitive_group_dataset is missing.
    with self.assertRaisesRegex(
        ValueError, "You must either.*or.*\n\n.*sensitive_group_dataset.*"
        "None.*\n.*nonsensitive_group_dataset.*Dataset.*\n"
        ".*min_diff_dataset.*Dataset"):
      _ = input_utils.pack_min_diff_data(
          inputs, nonsensitive_group_dataset=inputs, min_diff_dataset=inputs)

  def testInvalidStructureRaisesError(self):
    # Input dataset, content doesn't matter.
    inputs = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(5)

    nested_inputs = {"a": inputs, "b": inputs}
    bad_nested_inputs = {"a": inputs, "b": [inputs]}

    # No errors raised for valid nested structures.
    _ = input_utils.pack_min_diff_data(inputs, min_diff_dataset=nested_inputs)
    _ = input_utils.pack_min_diff_data(
        inputs,
        sensitive_group_dataset=nested_inputs,
        nonsensitive_group_dataset=nested_inputs)

    # Assert raises error for invalid min_diff_dataset structure.
    with self.assertRaisesRegex(
        ValueError, "min_diff_dataset.*unnested.*only "
        "elements of type.*Dataset.*Given"):
      _ = input_utils.pack_min_diff_data(
          inputs, min_diff_dataset=bad_nested_inputs)

    # Assert raises error for invalid sensitive_group_dataset structure.
    with self.assertRaisesRegex(
        ValueError, "sensitive_group_dataset.*unnested"
        ".*only elements of type.*Dataset.*Given"):
      _ = input_utils.pack_min_diff_data(
          inputs,
          sensitive_group_dataset=bad_nested_inputs,
          nonsensitive_group_dataset=nested_inputs)

    # Assert raises error for invalid nonsensitive_group_dataset structure.
    with self.assertRaisesRegex(
        ValueError, "nonsensitive_group_dataset.*unnested.*only elements of "
        "type.*Dataset.*Given"):
      _ = input_utils.pack_min_diff_data(
          inputs,
          sensitive_group_dataset=nested_inputs,
          nonsensitive_group_dataset=bad_nested_inputs)

    # Assert raises error for different sensitive and nonsensitive structures.
    different_nested_inputs = {"a": inputs, "c": inputs}
    with self.assertRaisesRegex(
        ValueError, "sensitive_group_dataset.*"
        "nonsensitive_group_dataset.*do "
        "not have the same structure(.|\n)*don't have the same set of keys"):
      _ = input_utils.pack_min_diff_data(
          inputs,
          sensitive_group_dataset=nested_inputs,
          nonsensitive_group_dataset=different_nested_inputs)

  def testDifferentMinDiffAndOriginalStructuresRaisesError(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], None, None)).batch(original_batch_size)

    sensitive_batch_sizes = [3, 5]
    sensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.sensitive_x, None, None)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], sensitive_batch_sizes)
    }

    nonsensitive_batch_sizes = [1, 2]
    nonsensitive_dataset = {
        key: tf.data.Dataset.from_tensor_slices(
            (self.nonsensitive_x, None, None)).batch(batch_size)
        for key, batch_size in zip(["k1", "k2"], nonsensitive_batch_sizes)
    }

    with self.assertRaisesRegex(
        ValueError, "x component structure.*min_diff_dataset.*does not match.*"
        "original x structure(.|\n)*don't have the same nested structure"):
      _ = input_utils.pack_min_diff_data(original_dataset, sensitive_dataset,
                                         nonsensitive_dataset)

  def testDifferentSensitiveAndNonsensitivetructuresRaisesError(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, None, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x["f1"], None, None)).batch(nonsensitive_batch_size)

    with self.assertRaisesRegex(
        ValueError, "x component structure.*sensitive_group_dataset.*does not "
        "match.*nonsensitive_group_dataset(.|\n)*don't have the same nested "
        "structure"):
      _ = input_utils.pack_min_diff_data(original_dataset, sensitive_dataset,
                                         nonsensitive_dataset)


class UnpackDataTest(tf.test.TestCase):

  def testUnpackOriginal(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.MinDiffPackedInputs(tensor, None)
    unpacked_tensor = input_utils.unpack_original_inputs(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

    # Dict of Tensors.
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    packed_inputs = input_utils.MinDiffPackedInputs(tensors, None)
    unpacked_tensors = input_utils.unpack_original_inputs(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.MinDiffPackedInputs(obj, None)
    unpacked_obj = input_utils.unpack_original_inputs(packed_inputs)
    self.assertIs(unpacked_obj, obj)

    # None.
    packed_inputs = input_utils.MinDiffPackedInputs(None, None)
    unpacked_obj = input_utils.unpack_original_inputs(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalDefaultsToIdentity(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_original_inputs(tensor)
    self.assertIs(tensor, unpacked_tensor)

    # Dict of Tensors.
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    unpacked_tensors = input_utils.unpack_original_inputs(tensors)
    self.assertIs(tensors, unpacked_tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_original_inputs(obj)
    self.assertIs(obj, unpacked_obj)

  def testUnpackMinDiffData(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.MinDiffPackedInputs(None, tensor)
    unpacked_tensor = input_utils.unpack_min_diff_data(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

    # Tuple of Tensors.
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    packed_inputs = input_utils.MinDiffPackedInputs(None, tensors)
    unpacked_tensors = input_utils.unpack_min_diff_data(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.MinDiffPackedInputs(None, obj)
    unpacked_obj = input_utils.unpack_min_diff_data(packed_inputs)
    self.assertIs(unpacked_obj, obj)

    # None.
    packed_inputs = input_utils.MinDiffPackedInputs(None, None)
    unpacked_obj = input_utils.unpack_min_diff_data(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackMinDiffDataDefaultsToNone(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_min_diff_data(tensor)
    self.assertIsNone(unpacked_tensor)

    # Tuple of Tensors.
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    unpacked_tensors = input_utils.unpack_min_diff_data(tensors)
    self.assertIsNone(unpacked_tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_min_diff_data(obj)
    self.assertIsNone(unpacked_obj)


if __name__ == "__main__":
  tf.test.main()
