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

"""Utils to process TensorFlow Model Analysis EvalResult into class structure.

Defines the following classes to compartmentalize various tfma.EvalResult data:
1. class MetricByFeatureSlice
2. class FeatureToSlices
3. class SliceVal
4. class SliceKey
"""

import dataclasses
from typing import Sequence, Union


# BEGIN EXPERIMENTAL-INTERNAL
# TODO: add a dictionary of valid metrics for the TFMR FDW
# library to lookup those which need to be transformed into loss within SliceVal
# or fdw_utils.
# END EXPERIMENTAL-INTERNAL
class SliceVal:
  """Stores feature-slice name and corresponding metric val as a loss.

  Note: metrics which are "naturally" losses are left as is. For metrics which
    follow the convention of higher the value, the better (e.g. 'accuracy'),
    these are subtracted from 1 to correspond to a loss representation.
  """

  def __init__(self, slice_name: str, metric_val_as_loss: float):
    """Instantiates a SliceVal object.

    Args:
      slice_name: the feature-slice identifier.
      metric_val_as_loss: the average slice-wise metric performance value.
    Note: every metric will be stored as a "loss", e.g. accuracy will be
      represented as (1 - metric).
    """
    self.slice_name = slice_name
    self.metric_val_as_loss = metric_val_as_loss


class FeatureToSlices:
  """Stores feature name (aka slice column) and performance on slices within this feature category."""

  def __init__(self, feature_name: str, slice_vals: Sequence[SliceVal]):
    """Instantiates a FeatureToSlice object.

    Args:
      feature_name: the feature column identifier.
      slice_vals: list of SliceVals for each feature-slice in feature_name.
    Note: assume that the feature_name exactly maps to what is in the tf.Example
    """
    self.feature_name = feature_name
    self.slice_vals = {}
    for slice_val in slice_vals:
      self.slice_vals[slice_val.slice_name] = slice_val


class MetricByFeatureSlice:
  """Stores metric name, feature name, and slices with corresponding metric val.

  Represents an object-oriented structure of the tfma.EvalResult with metric,
  feature, and slice performance information. Used in downstream tasks within
  Fair Data Reweighting library.
  """

  def __init__(self, metric_name: str, is_label_dependent: bool,
               dependent_label_value: Union[str, int, float],
               features_to_slices: Sequence[FeatureToSlices]):
    """Instantiates a MetricsByFeatureSlice object.

    Args:
      metric_name: the metric identifier. This maps exactly to what appears in
        the source tfma.EvalResult object
      is_label_dependent: a bool indicating if the metric is label dependent or
        independent.
      dependent_label_value: a string or numeric type indicating the label value
        the metric is dependent on.
      features_to_slices: list of FeatureToSlices, one per feature (column).
        Note that the .
    """
    self.metric_name = metric_name
    self.is_label_dependent = is_label_dependent
    self.dependent_label_value = dependent_label_value

    self.features_to_slices = {}
    for feature in features_to_slices:
      self.features_to_slices[feature.feature_name] = feature


@dataclasses.dataclass(frozen=True)
class SliceKey:
  """An object that represents a feature name and slice name."""

  feature_name: str
  slice_name: str

  def __str__(self) -> str:
    """Converts the SliceKey object to its string representation."""
    return f'{self.feature_name}_{self.slice_name}'

  def __bytes__(self) -> bytes:
    """Converts the SliceKey object to its byte representation."""
    return self.__str__().encode()
