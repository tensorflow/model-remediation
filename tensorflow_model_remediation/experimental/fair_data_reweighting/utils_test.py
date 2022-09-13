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

"""Tests for utils.py."""

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import FeatureToSlices
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import MetricByFeatureSlice
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceKey
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceVal
from tensorflow_model_remediation.experimental.fair_data_reweighting.test_util import serialize_example
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import get_slice_count
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import get_slice_keys_to_weights
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import has_key
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import infer_schema
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import tf_dataset_to_tf_examples_list

from google.protobuf import text_format


class SamplerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Generate small dummy data
    self.features = [
        [1.0, 0.3],
        [2.0, 0.4],
        [3.0, 0.5],
        [4.0, 0.6],
        [5.0, 0.7],
    ]

    self.labels = [0, 0, 1, 0, 1]
    self.slice_values = ["a", "b", "a", "c", "c"]

    slice_vals = [SliceVal("a", 0.3), SliceVal("b", 0.3), SliceVal("c", 0.3)]
    features_to_slices = FeatureToSlices("slice", slice_vals)

    self.eval_metrics = MetricByFeatureSlice(
        metric_name="loss",
        is_label_dependent=False,
        dependent_label_value=None,
        features_to_slices=[features_to_slices])

    self.slice_weight = {}
    self.slice_key_a = SliceKey("slice", "a")
    self.slice_key_b = SliceKey("slice", "b")
    self.slice_key_c = SliceKey("slice", "c")

    self.slice_weight[self.slice_key_a] = 1.0
    self.slice_weight[self.slice_key_b] = 1.0
    self.slice_weight[self.slice_key_c] = 1.0

    self.slice_count = {}
    self.slice_count[self.slice_key_a] = 2
    self.slice_count[self.slice_key_b] = 1
    self.slice_count[self.slice_key_c] = 2

    self.dataset_size_dict = {}
    self.dataset_size_dict[SliceKey("", "")] = 5

  def testHasKeyWithoutLabelDependence(self):
    tf_example = serialize_example(self.features[0][0], self.features[0][1],
                                   self.slice_values[0].encode("utf-8"),
                                   self.labels[0])
    flag = has_key(
        SliceKey("slice", "a"),
        tf_example,
        is_label_dependent=False,
        dependent_label_value=None,
        label_feature_column="label")
    self.assertEqual(flag, True)

  def testHasKeyWithLabelDependence(self):
    tf_example = serialize_example(self.features[0][0], self.features[0][1],
                                   self.slice_values[0].encode("utf-8"),
                                   self.labels[0])
    flag = has_key(
        SliceKey("slice", "a"),
        tf_example,
        is_label_dependent=True,
        dependent_label_value=1,
        label_feature_column="label")
    self.assertEqual(flag, False)

  def testHasKeyWithEmptySliceFeature(self):
    # create an input where `slice_with_empty_val` is a feature with no value.
    tf_example = text_format.Parse(
        """
        features {
          feature {
            key: "label"
            value {
              int64_list {
                value: 0
              }
            }
          }
          feature {
            key: "slice_with_empty_val"
            value {
              bytes_list {
              }
            }
          }
        }
        """, tf.train.Example())

    flag = has_key(
        SliceKey("slice_with_empty_val", "a"),
        tf_example,
        is_label_dependent=True,
        dependent_label_value=1,
        label_feature_column="label")
    self.assertEqual(flag, False)

  def testGetSlicedKeysToWeights(self):

    slice_keys_to_weights = get_slice_keys_to_weights(self.eval_metrics, 1.0)

    weights_dict = {}
    for key, _ in slice_keys_to_weights.items():
      weights_dict[key] = 1 / 3

    self.assertEqual(slice_keys_to_weights, weights_dict)

  def testGetSliceCount(self):

    slice_keys_to_weights = get_slice_keys_to_weights(self.eval_metrics, 1.0)
    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      slice_count = get_slice_count(dataset, list(slice_keys_to_weights.keys()),
                                    self.eval_metrics, "label", False)

      true_slice_count_dict = {}
      for key, _ in slice_keys_to_weights.items():
        if key.slice_name == "b":
          true_slice_count_dict[key] = 1
        else:
          true_slice_count_dict[key] = 2
      assert_that(slice_count, equal_to([true_slice_count_dict]))

  def testDatasetToTfExamplesMapper(self):
    dataset = tf.data.Dataset.from_tensor_slices({
        "features": self.features,
        "label": self.labels,
        "slice": self.slice_values
    })

    tf_dataset = list(tf_dataset_to_tf_examples_list(dataset))

    self.assertLen(tf_dataset, 5)

    features = []
    labels = []
    slices = []

    for point in tf_dataset:
      features.append(point.features.feature["features"].float_list.value)
      labels.append(point.features.feature["label"].int64_list.value[0])
      slices.append(
          point.features.feature["slice"].bytes_list.value[0].decode())

    self.assertListEqual(labels, self.labels)
    self.assertListEqual(slices, self.slice_values)

    # compare floating points individually
    self.assertEqual(len(features), len(self.features))
    for i in range(len(features)):
      self.assertEqual(len(features[i]), len(self.features[i]))
      for j in range(len(features[i])):
        self.assertAlmostEqual(features[i][j], self.features[i][j])

  def testSchemaInference(self):
    sparse_features = []
    for _ in range(len(self.labels)):
      sparse_features.append(
          tf.SparseTensor(
              indices=[[0, 0, 0], [0, 1, 2]],
              values=[1, 2],
              dense_shape=[1, 3, 4]))
    sparse_features_concat = tf.sparse.concat(axis=0, sp_inputs=sparse_features)

    dataset = tf.data.Dataset.from_tensor_slices({
        "features": self.features,
        "label": self.labels,
        "slice": self.slice_values,
        "sparse_features": sparse_features_concat
    })

    schema = infer_schema(dataset)

    self.assertSetEqual(
        set(schema.keys()),
        set(["features", "label", "slice", "sparse_features"]))
    self.assertEqual(schema["features"], tf.io.FixedLenFeature([2], tf.float32))
    self.assertEqual(schema["sparse_features"],
                     tf.io.FixedLenFeature([3, 4], tf.int64))
    self.assertEqual(schema["label"], tf.io.FixedLenFeature([], tf.int64))
    self.assertEqual(schema["slice"], tf.io.FixedLenFeature([], tf.string))


if __name__ == "__main__":
  tf.test.main()
