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

"""Util functions for the FDW algorithm."""

import math
from typing import Any, Dict, Iterator, List, Optional, Union

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import MetricByFeatureSlice
from tensorflow_model_remediation.experimental.fair_data_reweighting.datatypes import SliceKey


def get_first_value(example: tf.train.Example,
                    feature_name: str) -> Optional[Union[str, int, float]]:
  """Returns the first value of `feature_name` in `example`, or None if not found."""

  bundle = example.features.feature[feature_name]

  if bundle.HasField('bytes_list') and bundle.bytes_list.value:
    return bundle.bytes_list.value[0].decode('utf-8')
  elif bundle.HasField('float_list') and bundle.float_list.value:
    return bundle.float_list.value[0]
  elif bundle.HasField('int64_list') and bundle.int64_list.value:
    return bundle.int64_list.value[0]
  else:
    return None


def has_key(slice_key: SliceKey, example: tf.train.Example,
            is_label_dependent: bool, dependent_label_value: Union[str, int,
                                                                   float],
            label_feature_column: str) -> bool:
  """Finds if a tf example satisfies a given slice key.

  Arguments:
    slice_key: A SliceKey object.
    example: A given tf example.
    is_label_dependent: True if underyling metric is label dependent.
    dependent_label_value: The label value to match
    label_feature_column: The string representing the column name of the label.

  Returns:
    True if the example satisfies the slice.
  """

  if (slice_key.feature_name not in example.features.feature) or (
      slice_key.slice_name != get_first_value(example, slice_key.feature_name)):
    return False

  if (is_label_dependent) and (get_first_value(example, label_feature_column) !=
                               dependent_label_value):
    return False
  return True


def get_slice_keys_to_weights(eval_metrics: MetricByFeatureSlice,
                              beta: float) -> Dict[SliceKey, float]:
  """Computes the exponential weights for each SliceKey.

  Arguments:
    eval_metrics: the MetricByFeatureSlice object.
    beta: the temperature parameter in exponential weighting.

  Returns:
    A dictionary mapping _SliceKey objects to exponential weights.
  """
  total_sum = 0
  weights_dict = {}
  features_to_slices = eval_metrics.features_to_slices
  for feature_name, feature_to_slices in features_to_slices.items():
    slice_vals = feature_to_slices.slice_vals
    for slice_name, slice_val in slice_vals.items():
      val = math.exp(beta * slice_val.metric_val_as_loss)
      slice_key = SliceKey(feature_name, slice_name)
      weights_dict[slice_key] = val
      total_sum += val

  # Normalize by total_sum
  for slice_key, _ in weights_dict.items():
    weights_dict[slice_key] /= total_sum
  return weights_dict


def get_slice_count(
    examples: beam.PCollection[tf.train.Example], slice_keys: List[SliceKey],
    eval_metrics: MetricByFeatureSlice, label_feature_column: str,
    skip_slice_membership_check: bool) -> beam.PCollection[Dict[SliceKey, int]]:
  """Counts the number of examples for each slice_key.

  Arguments:
    examples: PCollection of all the examples.
    slice_keys: The list of slices.
    eval_metrics: The MetricByFeatureSlice object.
    label_feature_column: The column name of the label.
    skip_slice_membership_check: whether to check for slice membership or not.

  Returns:
    PCollection containing a Dict of slice keys to their counts.
  """

  def _count_by_slice_fn(example: tf.train.Example) -> Iterator[SliceKey]:
    """Yields `SliceKey` if the example matches the `SliceKey`.

    Arguments:
      example: The tf example.
    """
    for slice_key in slice_keys:
      if skip_slice_membership_check:
        yield slice_key
      elif has_key(slice_key, example, eval_metrics.is_label_dependent,
                   eval_metrics.dependent_label_value, label_feature_column):
        yield slice_key

  return (
      examples
      | f'Filter slices by {skip_slice_membership_check}' >>
      beam.FlatMap(_count_by_slice_fn)
      |
      f'Count slices with skip_slice_membership_check={skip_slice_membership_check}'
      >> beam.combiners.Count.PerElement()
      |
      f'To dictionary for skip_slice_membership_check={skip_slice_membership_check}'
      >> beam.combiners.ToDict())


def tf_dataset_to_tf_examples_list(
    data: tf.data.Dataset) -> Iterator[tf.train.Example]:
  """Convert tf.data.Dataset object to a list of tf.Example objects, to be used by beam.

  Arguments:
    data: a tf.data.Dataset dataset.

  Yields:
    Iterator of each dataset item in tf.train.Example format.
  """
  # BEGIN-GOOGLE-INTERNAL
  # Taken from
  # http://experimental3/third_party/py/seqio/utils.py;l=155;rcl=454506087.
  # END-GOOGLE-INTERNAL
  if not data:
    return []

  for elem in data.take(-1):
    np_elem = {k: v.numpy() for k, v in elem.items()}

    feature_dict = {}
    for k, v in np_elem.items():
      t = tf.constant(v)
      if len(t.shape) == 0:  # pylint:disable=g-explicit-length-test
        v = [v]
      elif len(t.shape) == 1:
        v = list(v)
      elif len(t.shape) >= 2:
        t = tf.reshape(v, [-1])
        v = v.flatten()

      if t.dtype == tf.string and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.compat.as_bytes(t) for t in v]))
      elif t.dtype in (tf.bool, tf.int32, tf.int64) and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=v))
      elif t.dtype in (tf.float32, tf.float64) and len(t.shape) <= 1:
        feature_dict[k] = tf.train.Feature(
            float_list=tf.train.FloatList(value=v))
      else:
        raise ValueError(
            "Unsupported type (%s) and shape (%s) for '%s' value: %s" %
            (t.dtype, t.shape, k, v))

    yield tf.train.Example(features=tf.train.Features(feature=feature_dict))


def infer_schema(data: tf.data.Dataset) -> Dict[str, Any]:
  """Infer the schema for the dataset.

  The schema is used when deserializing the dataset from TFRecords.
  This function determines the type and shape of each feature in the dataset,
  and creates a tf.io.FixedLenFeature of the type and shape.

  Arguments:
    data: tf.data.Dataset.

  Returns:
    A dictionary mapping each feature column of the dataset to an appropriate
    tf.io.FixedLenFeature.
  """

  def _normalize_dtype(dtype):
    """Normalize the dtype.

    Only three types (int64, float32, bytes) are supported while deserializing
    TFRecords.

    Arguments:
      dtype: initial data type.

    Returns:
      Normalized data type (one of int64, float32, bytes)
    """
    if dtype in (tf.int32, tf.int64, tf.bool):
      return tf.int64
    if dtype in (tf.float32, tf.float64):
      return tf.float32
    return dtype

  features_schema = {}

  for feature, spec in data.element_spec.items():
    features_schema[feature] = tf.io.FixedLenFeature(
        spec.shape, _normalize_dtype(spec.dtype))

  return features_schema
