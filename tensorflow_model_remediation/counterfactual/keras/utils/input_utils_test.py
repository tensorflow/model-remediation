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
    expected_counterfactual_data = tf.reshape(
        # Expected counterfactual dataset should not include the word "bad".
        tf.constant([" word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]),
        [25, 3])

    self.counterfactual_x = {
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2])),
        "f1_counterfactual": expected_counterfactual_data
    }
    self.counterfactual_w = tf.reshape(tf.range(275.0, 300.0), [25, 1])


class BuildCounterfactualDatasetTest(CounterfactualInputUtilsTestCase):

  def testBuildFromSingleDatasets(self):
    original_batch_size = 1
    original_input = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x["f1"], None,
             self.original_w)).batch(original_batch_size))

    counterfactual_list = ["bad"]

    dataset = input_utils.build_counterfactual_data(original_input,
                                                    counterfactual_list)

    expected_counterfactual_data = tf.reshape(
        tf.constant([" word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]), [25, 3])

    for batch_ind, counterfactual_batch in enumerate(dataset):
      # Assert counterfactual batch properly formed.
      counterfactual_x, counterfactual_y, counterfactual_w = (
          tf.keras.utils.unpack_x_y_sample_weight(counterfactual_batch))
      self.assertTensorsAllClose(
          counterfactual_x,
          _get_counterfactual_batch(self.original_x["f1"],
                                    original_batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_y,
          _get_counterfactual_batch(expected_counterfactual_data,
                                    original_batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_w,
          tf.ones_like(self.original_x["f1"], tf.float32))

  def testRegexForSensitiveWordsDoesWordMatch(self):
    original_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant([
            "He,", "here", " he.", " He .", "He is a doctor and she is a nurse"
        ]))
    counterfactual_list = ["he", "He", "she"]

    cf_dataset = input_utils.build_counterfactual_data(original_dataset,
                                                       counterfactual_list)
    expected_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant([",", "here", " .", "  .", " is a doctor and  is a nurse"]))

    for counterfactual_batch, original_batch, expected_batch in zip(
        cf_dataset, original_dataset, expected_dataset):
      original_x, counterfactual_x, _ = (
          tf.keras.utils.unpack_x_y_sample_weight(counterfactual_batch))
      self.assertTensorsAllClose(counterfactual_x, expected_batch)
      self.assertTensorsAllClose(original_x, original_batch)

  def testBuildFromCustomCounterfactualDatasets(self):
    original_batch_size = 1
    original_input = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))

    custom_feature_column = "f1"
    custom_replacement_dict = {"bad": "good"}

    def _create_counterfactual_data(original_input):
      original_x, original_y, original_sample_weight = (
          tf.keras.utils.unpack_x_y_sample_weight(original_input))
      for sensitive_word, new_word in custom_replacement_dict.items():
        counterfactual_data = tf.strings.regex_replace(
            original_x.get(custom_feature_column), sensitive_word, new_word)
        original_x[custom_feature_column +
                   "_counterfactual"] = counterfactual_data
      original_x.pop(custom_feature_column)
      return original_x, original_y, original_sample_weight

    dataset = input_utils.build_counterfactual_data(
        original_input,
        custom_counterfactual_function=_create_counterfactual_data)

    expected_counterfactual_data = tf.reshape(
        # Expected counterfactual dataset to replace "bad" with "good".
        tf.constant(["good word" + str(i) for i in range(25)] +
                    ["good word" + str(i) for i in range(50)]),
        [25, 3])
    expected_dict = {
        "f2": tf.reshape(tf.range(175.0, 225.0), [25, 2]),
        "f2_sparse": to_sparse(tf.reshape(tf.range(175.0, 225.0), [25, 2])),
        "f1_counterfactual": expected_counterfactual_data
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

  def testWithBothOriginalWeightsAreNotNone(self):
    original_batch_size = 3
    original_input = tf.data.Dataset.from_tensor_slices(
        (self.original_x["f1"], None, None)).batch(original_batch_size)

    counterfactual_list = ["bad"]

    dataset = input_utils.build_counterfactual_data(original_input,
                                                    counterfactual_list)

    # The resulting dataset will repeat infinitely so we only take the first 10
    # batches which corresponds to 2 full epochs of the counterfactual dataset.
    for counterfactual_batch in dataset.take(10):
      # Skip all counterfactual_data assertions except for weight.
      _, _, counterfactual_w = tf.keras.utils.unpack_x_y_sample_weight(
          counterfactual_batch)
      self.assertIsNotNone(counterfactual_w)

  def testCounterfactualDatasetsWithOnlyDatasetError(self):
    original_batch_size = 1
    original_input = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))
    with self.assertRaisesRegex(
        ValueError,
        "Either `custom_counterfactual_function` must be provided or .*\n"
        "Found:\nsensitive_terms_to_remove: None\n"
        "custom_counterfactual_function: None"):
      _ = input_utils.build_counterfactual_data(original_input)

  def testCounterfactualDatasetsWithOnlyFeatureColumnError(self):
    original_batch_size = 1
    original_input = (
        tf.data.Dataset.from_tensor_slices(
            (self.original_x, None,
             self.original_w)).batch(original_batch_size))
    with self.assertRaisesRegex(
        ValueError,
        "Either `custom_counterfactual_function` must be provided or .*\n"
        "Found:\nsensitive_terms_to_remove: None\n"
        "custom_counterfactual_function: None"):
      _ = input_utils.build_counterfactual_data(original_input)


class PackCounterfactualDataTest(CounterfactualInputUtilsTestCase):

  def testPackSingleDatasets(self):
    batch_size = 5
    original_input = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    counterfactual_data = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.counterfactual_x["f1_counterfactual"],
         tf.ones_like(self.original_x["f1"], tf.float32))).batch(batch_size)

    dataset = input_utils.pack_counterfactual_data(original_input,
                                                   counterfactual_data)

    for batch_ind, counterfactual_batch in enumerate(dataset):
      self.assertIsInstance(counterfactual_batch,
                            input_utils.CounterfactualPackedInputs)
      original_x, original_y, original_w = (
          tf.keras.utils.unpack_x_y_sample_weight(
              counterfactual_batch.original_input))
      original_x_in_counterfactual, counterfactual_x, counterfactual_w = (
          tf.keras.utils.unpack_x_y_sample_weight(
              counterfactual_batch.counterfactual_data))
      self.assertLen(counterfactual_batch, 2)
      self.assertTensorsAllClose(
          original_x,
          _get_batch(self.original_x, batch_size, batch_ind))
      self.assertTensorsAllClose(
          original_y,
          _get_batch(self.original_y, batch_size, batch_ind))
      self.assertIsNone(original_w)
      self.assertTensorsAllClose(
          original_x_in_counterfactual,
          _get_batch(self.original_x, batch_size, batch_ind))
      self.assertTensorsAllClose(
          counterfactual_x,
          _get_counterfactual_batch(self.counterfactual_x["f1_counterfactual"],
                                    batch_size, batch_ind))
      self.assertAllClose(
          counterfactual_w,
          tf.ones_like(counterfactual_x, tf.float32))

  def testCounterfactualDatasetIsRepeated(self):
    batch_size = 5
    original_input = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.original_y)).batch(batch_size)

    short_original_x = {
        "f1": tf.reshape(
            tf.constant(["bad word" + str(i) for i in range(30)]), [15, 2]),
        "f2": tf.reshape(tf.range(175.0, 205.0), [15, 2]),
    }

    short_counterfactual__dataset = tf.data.Dataset.from_tensor_slices(
        (short_original_x, None, None)).batch(batch_size)

    packed_dataset = input_utils.pack_counterfactual_data(
        original_input, short_counterfactual__dataset)
    self.assertEqual(packed_dataset.cardinality(), original_input.cardinality())

if __name__ == "__main__":
  tf.test.main()
