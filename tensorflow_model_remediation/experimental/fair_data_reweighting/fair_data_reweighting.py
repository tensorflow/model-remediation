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

"""Fair Data Reweighting remediation technique."""

import glob
import os
import tempfile
from typing import Any, Dict, Optional

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting import datatypes
from tensorflow_model_remediation.experimental.fair_data_reweighting import samplers
from tensorflow_model_remediation.experimental.fair_data_reweighting import utils

_REWEIGHTED_FILESET_NAME = 'reweighted'


def reweight_data(data: tf.data.Dataset,
                  eval_metrics: datatypes.MetricByFeatureSlice,
                  gamma: float,
                  beta: float,
                  label_feature_column: str,
                  dataset_schema_override: Optional[Dict[str, Any]] = None,
                  working_dir: Optional[str] = None) -> tf.data.Dataset:
  """Performs fair data reweighting on tf.data.Dataset.

  Arguments:
    data: A tf.data.Dataset to be reweighed.
    eval_metrics: The MetricByFeatureSlice object.
    gamma: The quantity governing the sampling factor.
    beta: The temperature parameter for expoential weighting.
    label_feature_column: The string representing the column name of the label.
    dataset_schema_override: Optionally give the schema, which is a dict mapping
      feature keys of `data` to tf.io.FixedLenFeature or tf.io.VarLenFeature
      values. If not given, it is automatically inferred from the dataset.
    working_dir: Optionally with a path to a working directory. If not, a temp
      directory will be used.

  Returns:
    The resampled dataset of the format tf.data.Dataset.
  """

  if not working_dir:
    working_dir = _get_temp_dir()

  if dataset_schema_override:
    dataset_schema = dataset_schema_override
  else:
    dataset_schema = utils.infer_schema(data)

  with beam.Pipeline() as p:
    # Create tf.Examples list as input
    tf_examples_data = (
        p | beam.Create(utils.tf_dataset_to_tf_examples_list(data)))

    # Reweight using beam implementation
    reweighted_data = reweight_pcollection(
        tf_examples_data,
        eval_metrics=eval_metrics,
        gamma=gamma,
        beta=beta,
        label_feature_column=label_feature_column)

    # Write to disk, because beam offers only file sinks
    _ = (
        reweighted_data | beam.io.WriteToTFRecord(
            os.path.join(working_dir, _REWEIGHTED_FILESET_NAME),
            coder=beam.coders.ProtoCoder(tf.train.Example)))

  # Read the reweighted dataset from disk
  reweighted_tf_dataset_raw = tf.data.TFRecordDataset(
      glob.glob(os.path.join(working_dir, _REWEIGHTED_FILESET_NAME + '*')))

  reweighted_tf_dataset = reweighted_tf_dataset_raw.map(
      lambda ex: tf.io.parse_single_example(ex, dataset_schema))

  return reweighted_tf_dataset


def reweight_pcollection(
    data: beam.PCollection[tf.train.Example],
    eval_metrics: datatypes.MetricByFeatureSlice, gamma: float, beta: float,
    label_feature_column: str) -> beam.PCollection[tf.train.Example]:
  """Performs fair data reweighting, for data represented as a PCollection.

  Arguments:
    data: A PCollection of tf examples.
    eval_metrics: The MetricByFeatureSlice object.
    gamma: The quantity governing the sampling factor.
    beta: The temperature parameter for exponential weighting.
    label_feature_column: The string representing the column name of the label.

  Returns:
    The resampled dataset.
  """
  # Get exponential weights for each slice
  slice_weight = utils.get_slice_keys_to_weights(eval_metrics, beta)

  # Get full dataset size
  slice_key = datatypes.SliceKey('', '')
  full_dataset_size_dict = utils.get_slice_count(
      data, [slice_key],
      eval_metrics,
      label_feature_column,
      skip_slice_membership_check=True)

  # Get counts for each slice
  slice_count = utils.get_slice_count(
      data,
      list(slice_weight.keys()),
      eval_metrics,
      label_feature_column,
      skip_slice_membership_check=False)

  # Get fair data distribution sampled with gamma factor.
  sampled_dataset = data | 'Sample Slices' >> beam.ParDo(
      samplers.SampleWithReplacement(),
      slice_weight=slice_weight,
      slice_count=beam.pvalue.AsSingleton(slice_count),
      eval_metrics=eval_metrics,
      gamma=gamma,
      dataset_size_dict=beam.pvalue.AsSingleton(full_dataset_size_dict),
      label_feature_column=label_feature_column,
      skip_slice_membership_check=False)

  # Get original data distribution sampled with 1-gamma factor.
  new_slice_weight = {}
  new_slice_weight[slice_key] = 1.0

  resampled_sliced_data = data | 'Sample Overall' >> beam.ParDo(
      samplers.SampleWithReplacement(),
      slice_weight=new_slice_weight,
      slice_count=beam.pvalue.AsSingleton(full_dataset_size_dict),
      eval_metrics=eval_metrics,
      gamma=1 - gamma,
      dataset_size_dict=beam.pvalue.AsSingleton(full_dataset_size_dict),
      label_feature_column=label_feature_column,
      skip_slice_membership_check=True)
  return (sampled_dataset,
          resampled_sliced_data) | 'Merge PCollections' >> beam.Flatten()


def _get_temp_dir():
  return tempfile.mkdtemp()
