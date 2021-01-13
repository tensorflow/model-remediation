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

"""Tests for input_util functions."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.keras.utils import input_utils


def _get_batch(tensors, batch_size, batch_num):
  if isinstance(tensors, dict):
    return {k: _get_batch(v, batch_size, batch_num) for k, v in tensors.items()}

  total_examples = len(tensors)
  start_ind = (batch_size * batch_num) % total_examples
  end_ind = start_ind + batch_size
  # Double tensor to enable repeating inputs.
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
        key: tf.concat([sensitive_batch[key], nonsensitive_batch[key]], axis=0)
        for key in sensitive_batch.keys()
    }
  return tf.concat([sensitive_batch, nonsensitive_batch], axis=0)


def _get_min_diff_membership_batch(sensitive_batch_size,
                                   nonsensitive_batch_size):
  return tf.concat(
      axis=0,
      values=[
          tf.ones([sensitive_batch_size, 1], tf.float32),
          tf.zeros([nonsensitive_batch_size, 1], tf.float32)
      ])


class PackMinDiffDataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Original inputs with 25 examples. Values go from 100.0 to 299.0.
    self.original_x = {
        "f1": tf.reshape(tf.range(100.0, 175.0), [25, 3]),
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
    }
    self.original_y = tf.reshape(tf.range(225.0, 275.0), [25, 2])
    self.original_w = tf.reshape(tf.range(275.0, 300.0), [25, 1])

    # Sensitive inputs with 15 examples. Values go from 300.0 to 399.0.
    self.sensitive_x = {
        "f1": tf.reshape(tf.range(300.0, 345.0), [15, 3]),
        "f2": tf.reshape(tf.range(345.0, 375.0), [15, 2]),
    }
    self.sensitive_w = tf.reshape(tf.range(375.0, 390.0), [15, 1])

    # Nonsensitive inputs with 10 examples. Values go from 400.0 to 499.0.
    self.nonsensitive_x = {
        "f1": tf.reshape(tf.range(400.0, 430.0), [10, 3]),
        "f2": tf.reshape(tf.range(430.0, 450.0), [10, 2]),
    }
    self.nonsensitive_w = tf.reshape(tf.range(450.0, 460.0), [10, 1])

  def testWithXAsTensor(self):

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

  def testWithXAsDict(self):
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

  def testWithOriginaleWeightsNone(self):
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
      _, _, w = tf.keras.utils.unpack_x_y_sample_weight(batch)

      # Skip original batch assertions except for weight.
      self.assertIsNone(w)

      # Skip all min_diff_data assertions.

  def testWithOnlySensitiveWeightsNone(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, None, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None,
         self.nonsensitive_w)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch_ind, batch in enumerate(dataset):
      packed_inputs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Skip original batch assertions.

      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(
          packed_inputs.min_diff_data)
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(
              tf.fill([sensitive_batch_size, 1], 1.0), self.nonsensitive_w,
              sensitive_batch_size, nonsensitive_batch_size, batch_ind))

  def testWithOnlyNonsensitiveWeightsNone(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, None, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, self.sensitive_w)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch_ind, batch in enumerate(dataset):
      packed_inputs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Skip original batch assertions.

      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(
          packed_inputs.min_diff_data)
      self.assertAllClose(
          min_diff_w,
          _get_min_diff_batch(self.sensitive_w,
                              tf.fill([nonsensitive_batch_size, 1],
                                      1.0), sensitive_batch_size,
                              nonsensitive_batch_size, batch_ind))

  def testWithBothMinDiffWeightsNone(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, None, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    dataset = input_utils.pack_min_diff_data(original_dataset,
                                             sensitive_dataset,
                                             nonsensitive_dataset)

    for batch in dataset:
      packed_inputs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
      self.assertIsInstance(packed_inputs, input_utils.MinDiffPackedInputs)

      # Skip original batch assertions.

      # Skip all min_diff_data assertions except for weight.
      _, _, min_diff_w = tf.keras.utils.unpack_x_y_sample_weight(
          packed_inputs.min_diff_data)
      self.assertIsNone(min_diff_w)

  def testDifferentWeightsShapeRaisesError(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, None, None)).batch(original_batch_size)

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
      _ = input_utils.pack_min_diff_data(original_dataset, sensitive_dataset,
                                         nonsensitive_dataset)

  def testDifferentMinDiffAndOriginalStructuresRaisesError(self):
    original_batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], None, None)).batch(original_batch_size)

    sensitive_batch_size = 3
    sensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.sensitive_x, None, None)).batch(sensitive_batch_size)

    nonsensitive_batch_size = 2
    nonsensitive_dataset = tf.data.Dataset.from_tensor_slices(
        (self.nonsensitive_x, None, None)).batch(nonsensitive_batch_size)

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
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

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
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
