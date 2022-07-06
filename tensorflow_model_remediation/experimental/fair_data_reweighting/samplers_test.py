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

"""Tests for Sampling with Replacement."""

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import FeatureToSlices
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import MetricByFeatureSlice
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceKey
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceVal
from tensorflow_model_remediation.experimental.fair_data_reweighting.samplers import SampleWithReplacement
from tensorflow_model_remediation.experimental.fair_data_reweighting.test_util import serialize_example


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

  def testSamplerWithGammaZero(self):
    # Should return an empty dataset
    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      sampled_dataset = dataset | beam.ParDo(
          SampleWithReplacement(),
          slice_weight=self.slice_weight,
          slice_count=self.slice_count,
          eval_metrics=self.eval_metrics,
          gamma=0.0,
          dataset_size_dict=self.dataset_size_dict,
          label_feature_column="label",
          skip_slice_membership_check=False)

      n = sampled_dataset | beam.combiners.Count.Globally()
      assert_that(n, equal_to([0]))

  def testSamplerWithGammaNonZero(self):
    # Should return 5 points from slice_b and 4 to 6 points each from slice_a,
    # slice_b
    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      _ = dataset | beam.ParDo(
          SampleWithReplacement(),
          slice_weight=self.slice_weight,
          slice_count=self.slice_count,
          eval_metrics=self.eval_metrics,
          gamma=1.0,
          dataset_size_dict=self.dataset_size_dict,
          label_feature_column="label",
          skip_slice_membership_check=False)

      pipeline.run()
    counters = pipeline.result.metrics().query()["counters"]
    counts = {
        counter.key.metric.name: counter.committed for counter in counters
    }

    self.assertEqual(counts[str(self.slice_key_b)], 5)
    self.assertGreater(counts[str(self.slice_key_a)], 3)
    self.assertLess(counts[str(self.slice_key_a)], 7)
    self.assertGreater(counts[str(self.slice_key_c)], 3)
    self.assertLess(counts[str(self.slice_key_c)], 7)

  def testSamplerWithLabelDependence(self):
    # Should return 5 points from each slice

    self.eval_metrics.is_label_dependent = True
    self.eval_metrics.dependent_label_value = 0
    self.slice_count[self.slice_key_a] = 1
    self.slice_count[self.slice_key_b] = 1
    self.slice_count[self.slice_key_c] = 1

    with beam.Pipeline() as pipeline:
      dataset = (
          pipeline
          | beam.Create([
              serialize_example(self.features[i][0], self.features[i][1],
                                self.slice_values[i].encode("utf-8"),
                                self.labels[i]) for i in range(5)
          ]))

      sampled_dataset = dataset | beam.ParDo(
          SampleWithReplacement(),
          slice_weight=self.slice_weight,
          slice_count=self.slice_count,
          eval_metrics=self.eval_metrics,
          gamma=1.0,
          dataset_size_dict=self.dataset_size_dict,
          label_feature_column="label",
          skip_slice_membership_check=False)

      n = sampled_dataset | beam.combiners.Count.Globally()
      assert_that(n, equal_to([15]))


if __name__ == "__main__":
  tf.test.main()
