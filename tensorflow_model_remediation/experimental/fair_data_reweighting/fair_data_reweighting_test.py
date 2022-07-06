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

"""Tests for Fair Data Reweighting technique."""

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import numpy as np
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting import fair_data_reweighting as fdw
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import FeatureToSlices
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import MetricByFeatureSlice
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceKey
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceVal
from tensorflow_model_remediation.experimental.fair_data_reweighting.test_util import serialize_example


class FairDataReweightingTest(tf.test.TestCase):

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

  def testReweightDataWithGammaZero(self):
    # Should get back original dataset
    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      reweighted_dataset = fdw.reweight_pcollection(dataset, self.eval_metrics,
                                                    0.0, 1.0, "label")

      n = reweighted_dataset | beam.combiners.Count.Globally()
      assert_that(n, equal_to([5]))

  def testReweightDataWithGammaOne(self):
    # Should sample 2 points per slice
    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      reweighted_dataset = fdw.reweight_pcollection(dataset, self.eval_metrics,
                                                    1.0, 1.0, "label")

      n = reweighted_dataset | beam.combiners.Count.Globally()
      assert_that(n, equal_to([6]))

  def testReweightDataWithGammaHalf(self):
    total_counts = {}
    total_counts[str(self.slice_key_a)] = []
    total_counts[str(self.slice_key_b)] = []
    total_counts[str(self.slice_key_c)] = []
    # Use empty SliceKey to count the data points sampled from the full dataset.
    total_counts[str(SliceKey("", ""))] = []
    # Should sample 2 points (on average) from the overall dataset and,
    # One point (on average) per slice
    for _ in range(10):
      with beam.Pipeline() as pipeline:
        dataset = (
            pipeline
            | beam.Create([
                serialize_example(self.features[i][0], self.features[i][1],
                                  self.slice_values[i].encode("utf-8"),
                                  self.labels[i]) for i in range(5)
            ]))

        _ = fdw.reweight_pcollection(dataset, self.eval_metrics, 0.5, 1.0,
                                     "label")
        pipeline.run()
      counters = pipeline.result.metrics().query()["counters"]
      counts = {
          counter.key.metric.name: counter.committed for counter in counters
      }
      if str(self.slice_key_a) in counts:
        total_counts[str(self.slice_key_a)].append(counts[str(
            self.slice_key_a)])
      else:
        total_counts[str(self.slice_key_a)].append(0)
      if str(self.slice_key_b) in counts:
        total_counts[str(self.slice_key_b)].append(counts[str(
            self.slice_key_b)])
      else:
        total_counts[str(self.slice_key_b)].append(0)
      if str(self.slice_key_c) in counts:
        total_counts[str(self.slice_key_c)].append(counts[str(
            self.slice_key_c)])
      else:
        total_counts[str(self.slice_key_c)].append(0)
      if str(SliceKey("", "")) in counts:
        total_counts[str(SliceKey("", ""))].append(counts[str(SliceKey("",
                                                                       ""))])
      else:
        total_counts[str(SliceKey("", ""))].append(0)

    self.assertAlmostEqual(
        np.mean(total_counts[str(SliceKey("", ""))]), 2, delta=2)
    self.assertAlmostEqual(
        np.mean(total_counts[str(self.slice_key_a)]), 1, delta=1)
    self.assertAlmostEqual(
        np.mean(total_counts[str(self.slice_key_b)]), 1, delta=1)
    self.assertAlmostEqual(
        np.mean(total_counts[str(self.slice_key_c)]), 1, delta=1)

  def testReweightDataWithLabelDependence(self):

    self.eval_metrics.is_label_dependent = True
    self.eval_metrics.dependent_label_value = 1

    total_counts = {}
    total_counts[str(self.slice_key_a)] = []
    total_counts[str(self.slice_key_b)] = []
    total_counts[str(self.slice_key_c)] = []
    total_counts[str(SliceKey("", ""))] = []
    # Should sample 2 points (on average) from the overall dataset and,
    # One point (on average) per slices 'a' and 'c'
    for _ in range(10):
      with beam.Pipeline() as pipeline:
        dataset = (
            pipeline
            | beam.Create([
                serialize_example(self.features[i][0], self.features[i][1],
                                  self.slice_values[i].encode("utf-8"),
                                  self.labels[i]) for i in range(5)
            ]))

        _ = fdw.reweight_pcollection(dataset, self.eval_metrics, 0.5, 1.0,
                                     "label")
        pipeline.run()
      counters = pipeline.result.metrics().query()["counters"]
      counts = {
          counter.key.metric.name: counter.committed for counter in counters
      }
      if str(self.slice_key_a) in counts:
        total_counts[str(self.slice_key_a)].append(counts[str(
            self.slice_key_a)])
      else:
        total_counts[str(self.slice_key_a)].append(0)
      if str(self.slice_key_b) in counts:
        total_counts[str(self.slice_key_b)].append(counts[str(
            self.slice_key_b)])
      else:
        total_counts[str(self.slice_key_b)].append(0)
      if str(self.slice_key_c) in counts:
        total_counts[str(self.slice_key_c)].append(counts[str(
            self.slice_key_c)])
      else:
        total_counts[str(self.slice_key_c)].append(0)
      if str(SliceKey("", "")) in counts:
        total_counts[str(SliceKey("", ""))].append(counts[str(SliceKey("",
                                                                       ""))])
      else:
        total_counts[str(SliceKey("", ""))].append(0)

    self.assertAlmostEqual(
        np.mean(total_counts[str(SliceKey("", ""))]), 2, delta=2)
    self.assertAlmostEqual(
        np.mean(total_counts[str(self.slice_key_a)]), 1, delta=1)
    self.assertEqual(np.mean(total_counts[str(self.slice_key_b)]), 0)
    self.assertAlmostEqual(
        np.mean(total_counts[str(self.slice_key_c)]), 1, delta=1)

  def testReweightingWithTfDataset(self):
    dataset = tf.data.Dataset.from_tensor_slices({
        "features": self.features,
        "label": self.labels,
        "slice": self.slice_values
    })

    schema = {
        "features": tf.io.FixedLenFeature([2], tf.float32),
        "slice": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    reweighted_dataset = fdw.reweight_data(dataset, self.eval_metrics, 1.0, 1.0,
                                           "label", schema)
    sz = len(list(reweighted_dataset.as_numpy_iterator()))
    self.assertEqual(sz, 6)

  def testReweightingWithTfDatasetAndInferredSchema(self):
    dataset = tf.data.Dataset.from_tensor_slices({
        "features": self.features,
        "label": self.labels,
        "slice": self.slice_values
    })

    reweighted_dataset = fdw.reweight_data(dataset, self.eval_metrics, 1.0, 1.0,
                                           "label")
    sz = len(list(reweighted_dataset.as_numpy_iterator()))
    self.assertEqual(sz, 6)


if __name__ == "__main__":
  tf.test.main()
