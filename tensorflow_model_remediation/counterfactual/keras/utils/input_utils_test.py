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

"""Tests for input_utils."""

import tensorflow as tf

from tensorflow_model_remediation.counterfactual.keras.utils import input_utils


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


def _get_counterfactual_batch(counterfactual_tensors, batch_size, batch_num):
  counterfactual_batch = _get_batch(
      counterfactual_tensors, batch_size, batch_num)
  if isinstance(counterfactual_batch, dict):
    return {
        key: counterfactual_batch[key]
        for key in counterfactual_batch.keys()
    }
  return counterfactual_batch


def to_sparse(tensor):
  """Helper to create a SparseTensor from a dense Tensor."""
  values = tf.reshape(tensor, [-1])
  indices = tf.where(tensor)
  shape = tf.shape(tensor, out_type=tf.int64)
  return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


class CounterfactualInputUtilsTestCase(tf.test.TestCase):

  def assertTensorsAllClose(self, a, b):
    """recursive comparison to handle SparseTensors."""
    if isinstance(a, dict):
      for key, a_value in a.items():
        b_value = b.get(key)
        self.assertIsNotNone(b_value)
        self.assertTensorsAllClose(a_value, b_value)
    elif isinstance(a, tf.SparseTensor):
      self.assertAllEqual(a.indices, b.indices)
      self.assertAllClose(a.values, b.values)
      self.assertTensorsAllClose(a.dense_shape, b.dense_shape)
    else:
      if tf.is_tensor(a):
        self.assertIsNone(tf.debugging.assert_equal(a, b))
      else:
        self.assertAllClose(a, b)

  def setUp(self):
    super().setUp()

    # Original inputs with 25 examples.
    self.original_x = {
        "f1": tf.reshape(
            tf.constant(["bad word" + str(i) for i in range(25)] +
                        ["good word" + str(i) for i in range(50)]), [25, 3]),
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2]))
    }
    self.original_y = tf.reshape(tf.range(225.0, 275.0), [25, 2])
    self.original_w = tf.reshape(tf.range(275.0, 300.0), [25, 1])

    # Counterfactual inputs with 25 examples.
    expected_counterfactual_dataset = tf.reshape(
        # Expected counterfactual dataset should not include the word "bad".
        tf.constant([" word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]),
        [25, 3])

    self.counterfactual_x = {
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2])),
        "f1_counterfactual": expected_counterfactual_dataset
    }
    self.counterfactual_w = tf.reshape(tf.range(275.0, 300.0), [25, 1])


class BuildCounterfactualDatasetTest(CounterfactualInputUtilsTestCase):

  def testBuildFromSingleDatasets(self):
    original_batch_size = 1
    original_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x["f1"], None,
             self.original_w)).batch(original_batch_size))

    counterfactual_list = ["bad"]

    dataset = input_utils.build_counterfactual_dataset(original_dataset,
                                                       counterfactual_list)

    expected_counterfactual_dataset = tf.reshape(
        # Expected counterfactual dataset should not include the word "bad".
        tf.constant([" word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]),
        [25, 3])

    for batch_ind, counterfactual_batch in enumerate(dataset):
      # Assert counterfactual batch properly formed.
      counterfactual_x, counterfactual_y, counterfactual_w = (
          tf.keras.utils.unpack_x_y_sample_weight(counterfactual_batch))
      self.assertTensorsAllClose(
          counterfactual_x,
          _get_counterfactual_batch(expected_counterfactual_dataset,
                                    original_batch_size, batch_ind))
      self.assertIsNone(counterfactual_y)
      self.assertIsNone(counterfactual_w)

  def testBuildFromCustomCounterfactualDatasets(self):
    original_batch_size = 1
    original_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))

    custom_feature_column = "f1"
    custom_replacement_dict = {"bad": "good"}

    def _create_counterfactual_dataset(original_dataset):
      original_x, original_y, original_sample_weight = (
          tf.keras.utils.unpack_x_y_sample_weight(original_dataset))
      for sensitive_word, new_word in custom_replacement_dict.items():
        counterfactual_dataset = tf.strings.regex_replace(
            original_x.get(custom_feature_column), sensitive_word, new_word)
        original_x[custom_feature_column +
                   "_counterfactual"] = counterfactual_dataset
      original_x.pop(custom_feature_column)
      return original_x, original_y, original_sample_weight

    dataset = input_utils.build_counterfactual_dataset(
        original_dataset,
        custom_counterfactual_function=_create_counterfactual_dataset)

    expected_counterfactual_dataset = tf.reshape(
        # Expected counterfactual dataset to replace "bad" with "good".
        tf.constant(["good word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]),
        [25, 3])
    expected_dict = {
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2])),
        "f1_counterfactual": expected_counterfactual_dataset
    }
    self.assertListEqual(list(expected_dict.keys()),
                         list(self.counterfactual_x.keys()))

    for batch_ind, counterfactual_batch in enumerate(dataset):
      # Assert counterfactual batch properly formed.
      counterfactual_x, counterfactual_y, counterfactual_w = (
          counterfactual_batch)
      self.assertTensorsAllClose(
          counterfactual_x,
          _get_counterfactual_batch(expected_dict,
                                    original_batch_size, batch_ind))
      self.assertIsNone(counterfactual_y)
      self.assertTensorsAllClose(
          counterfactual_w,
          _get_counterfactual_batch(self.original_w,
                                    original_batch_size, batch_ind))

  # TODO: Add tests that the original and counterfactual appear within
  # the same batch only.

  def testWithBothOriginalWeightsNone(self):
    original_batch_size = 3
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], None, None)).batch(original_batch_size)

    counterfactual_list = ["bad"]

    dataset = input_utils.build_counterfactual_dataset(original_dataset,
                                                       counterfactual_list)

    # The resulting dataset will repeat infinitely so we only take the first 10
    # batches which corresponds to 2 full epochs of the counterfactual dataset.
    for counterfactual_batch in dataset.take(10):
      # Skip all counterfactual_data assertions except for weight.
      _, _, counterfactual_w = tf.keras.utils.unpack_x_y_sample_weight(
          counterfactual_batch)
      self.assertIsNone(counterfactual_w)

  def testCounterfactualDatasetsWithOnlyDatasetError(self):
    original_batch_size = 1
    original_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))
    with self.assertRaisesRegex(
        ValueError,
        "Either `custom_counterfactual_function` must be provided or .*\n"
        "Found:\nsensitive_terms_to_remove: None\n"
        "custom_counterfactual_function: None"):
      _ = input_utils.build_counterfactual_dataset(original_dataset)

  def testCounterfactualDatasetsWithOnlyFeatureColumnError(self):
    original_batch_size = 1
    original_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))
    with self.assertRaisesRegex(
        ValueError,
        "Either `custom_counterfactual_function` must be provided or .*\n"
        "Found:\nsensitive_terms_to_remove: None\n"
        "custom_counterfactual_function: None"):
      _ = input_utils.build_counterfactual_dataset(original_dataset)


class PackCounterfactualDataTest(CounterfactualInputUtilsTestCase):

  def testPackSingleDatasets(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], self.original_y,
         self.original_w)).batch(batch_size)

    counterfactual_dataset = tf.data.Dataset.from_tensor_slices(
        (self.counterfactual_x["f1_counterfactual"], None,
         self.counterfactual_w)).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(original_dataset,
                                                   counterfactual_dataset)

    for batch_ind, counterfactual_packed_inputs in enumerate(dataset):
      self.assertIsInstance(counterfactual_packed_inputs,
                            input_utils.CounterfactualPackedInputs)
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_x,
          _get_batch(self.original_x["f1"], batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_y,
          _get_batch(self.original_y, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_sample_weight,
          _get_batch(self.original_w, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.counterfactual_x,
          _get_counterfactual_batch(self.counterfactual_x["f1_counterfactual"],
                                    batch_size, batch_ind))
      self.assertAllClose(
          counterfactual_packed_inputs.counterfactual_sample_weight,
          _get_counterfactual_batch(self.counterfactual_w,
                                    batch_size, batch_ind))

  def testPackSingleDatasetsWithDict(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y,
         self.original_w)).batch(batch_size)

    counterfactual_dataset = tf.data.Dataset.from_tensor_slices(
        (self.counterfactual_x, None, self.counterfactual_w)).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(original_dataset,
                                                   counterfactual_dataset)

    for batch_ind, counterfactual_packed_inputs in enumerate(dataset):
      self.assertIsInstance(counterfactual_packed_inputs,
                            input_utils.CounterfactualPackedInputs)
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_x,
          _get_batch(self.original_x, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_y,
          _get_batch(self.original_y, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.original_sample_weight,
          _get_batch(self.original_w, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.counterfactual_x,
          _get_counterfactual_batch(self.counterfactual_x,
                                    batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_packed_inputs.counterfactual_sample_weight,
          _get_counterfactual_batch(self.counterfactual_w,
                                    batch_size, batch_ind))

  def testWithoutOriginalWeights(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    counterfactual_dataset = tf.data.Dataset.from_tensor_slices(
        (self.counterfactual_x["f1_counterfactual"], None,
         self.counterfactual_w)).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(original_dataset,
                                                   counterfactual_dataset)

    for batch in dataset:
      # Only validate original batch weights (other tests cover others).
      # Should be of length 5.
      self.assertLen(batch, 5)
      self.assertIsNone(batch.original_sample_weight)

  def testWithoutCounterfactualWeights(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    counterfactual_dataset = tf.data.Dataset.from_tensor_slices(
        (self.counterfactual_x["f1_counterfactual"], None,
         None)).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(original_dataset,
                                                   counterfactual_dataset)

    for batch in dataset:
      self.assertLen(batch, 5)
      self.assertIsNone(batch.counterfactual_sample_weight)

  def testWithParameterCounterfactualWeights(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    counterfactual_dataset = tf.data.Dataset.from_tensor_slices(
        (self.counterfactual_x["f1_counterfactual"], None,
         None)).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(
        original_dataset,
        counterfactual_dataset,
        cf_sample_weight=2.0)
    expected_counterfactual_sample_weight = tf.fill([25, 1], 2.0)

    for batch_ind, batch in enumerate(dataset):
      self.assertLen(batch, 5)
      self.assertTensorsAllClose(
          batch.counterfactual_sample_weight,
          _get_counterfactual_batch(expected_counterfactual_sample_weight,
                                    batch_size, batch_ind))

  def testInvalidStructureRaisesError(self):
    batch_size = 5
    original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    short_original_x = {
        "f1": tf.reshape(
            tf.constant(["bad word" + str(i) for i in range(30)]), [15, 2]),
        "f2": tf.reshape(tf.range(175.0, 205.0), [15, 2]),
    }

    short_counterfactual__dataset = tf.data.Dataset.from_tensor_slices(
        (short_original_x, None, None)).batch(batch_size)

    with self.assertRaisesRegex(
        ValueError, ".*Original cardinality: 5\nCounterfactual cardinality: 3"):
      _ = input_utils.pack_counterfactual_data(
          original_dataset, short_counterfactual__dataset)


class UnpackDataTest(tf.test.TestCase):

  def testUnpackOriginalX(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.CounterfactualPackedInputs(
        tensor, None, None, None, None)
    unpacked_tensor = input_utils.unpack_original_x(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

    # Dict of Tensors.
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    packed_inputs = input_utils.CounterfactualPackedInputs(
        tensors, None, None, None, None)
    unpacked_tensors = input_utils.unpack_original_x(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.CounterfactualPackedInputs(
        obj, None, None, None, None)
    unpacked_obj = input_utils.unpack_original_x(packed_inputs)
    self.assertIs(unpacked_obj, obj)

    # None.
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, None)
    unpacked_obj = input_utils.unpack_original_x(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalY(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, tensor, None, None, None)
    unpacked_tensor = input_utils.unpack_original_y(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

    # Tuple of Tensors.
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, tensors, None, None, None)
    unpacked_tensors = input_utils.unpack_original_y(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, obj, None, None, None)
    unpacked_obj = input_utils.unpack_original_y(packed_inputs)
    self.assertIs(unpacked_obj, obj)

    # None.
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, None)
    unpacked_obj = input_utils.unpack_original_y(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalSampleWeight(self):
    # Tensor.
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, tensor, None, None)
    unpacked_tensor = input_utils.unpack_original_sample_weight(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

    # Tuple of Tensors.
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, tensors, None, None)
    unpacked_tensors = input_utils.unpack_original_sample_weight(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

    # Arbitrary object.
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, obj, None, None)
    unpacked_obj = input_utils.unpack_original_sample_weight(packed_inputs)
    self.assertIs(unpacked_obj, obj)

    # None.
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, None)
    unpacked_obj = input_utils.unpack_original_y(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackCounterfactualXTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, tensor, None)
    unpacked_tensor = input_utils.unpack_counterfactual_x(packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

  def testUnpackCounterfactualXTupleOfTensor(self):
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, tensors, None)
    unpacked_tensors = input_utils.unpack_counterfactual_x(packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

  def testUnpackCounterfactualXObject(self):
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, obj, None)
    unpacked_obj = input_utils.unpack_counterfactual_x(packed_inputs)
    self.assertIs(unpacked_obj, obj)

  def testUnpackCounterfactualXNone(self):
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, None)
    unpacked_obj = input_utils.unpack_counterfactual_x(packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalXDefaultsToNoneTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_original_x(tensor)
    self.assertIsNone(unpacked_tensor)

  def testUnpackOriginalXDefaultsToINoneDictOfTensor(self):
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    unpacked_tensors = input_utils.unpack_original_x(tensors)
    self.assertIsNone(unpacked_tensors)

  def testUnpackOriginalXDefaultsToNoneobject(self):
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_original_x(obj)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalYDefaultsToNoneTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_original_y(tensor)
    self.assertIsNone(unpacked_tensor)

  def testUnpackOriginalYDefaultsToNoneDictOfTensor(self):
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    unpacked_tensors = input_utils.unpack_original_y(tensors)
    self.assertIsNone(unpacked_tensors)

  def testUnpackOriginalYDefaultsToNoneObject(self):
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_original_y(obj)
    self.assertIsNone(unpacked_obj)

  def testUnpackOriginalSampleWeightDefaultsToNoneTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_original_sample_weight(tensor)
    self.assertIsNone(unpacked_tensor)

  def testUnpackOriginalSampleWeightDefaultsToNoneDictOfTensor(self):
    tensors = {"f1": tf.fill([1, 2], 2), "f2": tf.fill([4, 5], "a")}
    unpacked_tensors = input_utils.unpack_original_sample_weight(tensors)
    self.assertIsNone(unpacked_tensors)

  def testUnpackOriginalSampleWeightDefaultsToNoneObject(self):
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_original_sample_weight(obj)
    self.assertIsNone(unpacked_obj)

  def testUnpackCounterfactualSampleWeightTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, tensor)
    unpacked_tensor = input_utils.unpack_counterfactual_sample_weight(
        packed_inputs)
    self.assertIs(unpacked_tensor, tensor)

  def testUnpackCounterfactualSampleWeightTupleOfTensor(self):
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, tensors)
    unpacked_tensors = input_utils.unpack_counterfactual_sample_weight(
        packed_inputs)
    self.assertIs(unpacked_tensors, tensors)

  def testUnpackCounterfactualSampleWeightObject(self):
    obj = set(["a", "b", "c"])
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, obj)
    unpacked_obj = input_utils.unpack_counterfactual_sample_weight(
        packed_inputs)
    self.assertIs(unpacked_obj, obj)

  def testUnpackCounterfactualSampleWeightNone(self):
    packed_inputs = input_utils.CounterfactualPackedInputs(
        None, None, None, None, None)
    unpacked_obj = input_utils.unpack_counterfactual_sample_weight(
        packed_inputs)
    self.assertIsNone(unpacked_obj)

  def testUnpackCounterfactualXToNoneTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_counterfactual_x(tensor)
    self.assertIsNone(unpacked_tensor)

  def testUnpackCounterfactualXToNoneTupleOfTensor(self):
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    unpacked_tensors = input_utils.unpack_counterfactual_x(tensors)
    self.assertIsNone(unpacked_tensors)

  def testUnpackCounterfactualXToNoneObject(self):
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_counterfactual_x(obj)
    self.assertIsNone(unpacked_obj)

  def testUnpackCounterfactualSampleWeightToNoneTensor(self):
    tensor = tf.fill([3, 4], 1.3)
    unpacked_tensor = input_utils.unpack_counterfactual_sample_weight(tensor)
    self.assertIsNone(unpacked_tensor)

  def testUnpackCounterfactualSampleWeightToNoneTupleOfTensor(self):
    tensors = ({
        "f1": tf.fill([1, 2], 2),
        "f2": tf.fill([4, 5], "a")
    }, None, tf.fill([4, 1], 1.0))
    unpacked_tensors = input_utils.unpack_counterfactual_sample_weight(tensors)
    self.assertIsNone(unpacked_tensors)

  def testUnpackCounterfactualSampleWeightToNoneObject(self):
    obj = set(["a", "b", "c"])
    unpacked_obj = input_utils.unpack_counterfactual_sample_weight(obj)
    self.assertIsNone(unpacked_obj)

  def testUnpackXySampleWeightCfxCfsampleWeightValueError(self):
    with self.assertRaises(TypeError):
      _ = input_utils.unpack_x_y_sample_weight_cfx_cfsample_weight("wrong type")


if __name__ == "__main__":
  tf.test.main()
