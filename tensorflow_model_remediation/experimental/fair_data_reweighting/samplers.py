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

"""(Approximate) Sampling with replacement algorithms."""

import math
import random
from typing import Dict, Iterator

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import MetricByFeatureSlice
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceKey
from tensorflow_model_remediation.experimental.fair_data_reweighting.utils import has_key


def _get_random_number() -> np.random.Generator:
  """Returns a default random number generator."""
  return random.uniform(0, 1.0)


class SampleWithReplacement(beam.DoFn):
  """Approximate Sampling with replacement.

  If k are desired from a slice with m total points, the algorithm first
  replicates each point in the slice floor(k/m) points. In the second step, the
  remaining points (if any) are drawn independently at random via a single pass
  over the slice with sampling probability of 1/m.
  """

  def process(self, example: tf.train.Example,
              slice_weight: Dict[SliceKey, float], slice_count: Dict[SliceKey,
                                                                     int],
              eval_metrics: MetricByFeatureSlice, gamma: float,
              dataset_size_dict: Dict[SliceKey, int], label_feature_column: str,
              skip_slice_membership_check: bool) -> Iterator[tf.train.Example]:
    """Replicates a given example an appropriate number of times.

    If k points are desired from a slice with m total points, the replication
    factor is floor(k/m).

    Arguments:
      example: A given tf example.
      slice_weight: A dict mapping slice keys to weights
      slice_count: A dict mapping slice keys to counts
      eval_metrics: The eval_metrics object
      gamma: The quantity governing the sampling factor
      dataset_size_dict: The size of the full train dataset
      label_feature_column: The column name of the label.
      skip_slice_membership_check: whether to check for slice membership or not.

    Yields:
      The example replicated an appropriate number of times.
    """
    for slice_key, weight in slice_weight.items():
      if (skip_slice_membership_check) or has_key(
          slice_key, example, eval_metrics.is_label_dependent,
          eval_metrics.dependent_label_value, label_feature_column):
        slice_num_points = slice_count[slice_key]
        dataset_size = list(dataset_size_dict.items())[0][1]

        # Sample (on average) num_points_to_sample total points from the slice
        num_points_to_sample = math.ceil(gamma * weight * dataset_size)

        # Compute # times a data point needs to be replicated deterministically
        rep_factor = int(num_points_to_sample / slice_num_points)

        # Compute remaining_points to be sampled randomly
        remaining_points = num_points_to_sample - rep_factor * slice_num_points

        # Deterministically replicate the point rep_factor times
        for _ in range(rep_factor):
          yield example
          beam.metrics.Metrics.counter('slice_sample_count',
                                       str(slice_key)).inc()

        # Random sampling with probability=remaining_points/slice_num_points
        if _get_random_number() <= remaining_points / slice_num_points:
          yield example
          beam.metrics.Metrics.counter('slice_sample_count',
                                       str(slice_key)).inc()
