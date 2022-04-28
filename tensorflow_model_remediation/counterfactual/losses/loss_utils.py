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

"""Utility functions for Counterfactual losses."""

from typing import Union

from tensorflow_model_remediation.counterfactual.losses import base_loss
from tensorflow_model_remediation.counterfactual.losses import pairwise_absolute_difference_loss
from tensorflow_model_remediation.counterfactual.losses import pairwise_cosine_loss
from tensorflow_model_remediation.counterfactual.losses import pairwise_mse_loss

_STRING_TO_LOSS_DICT = {}


def _register_loss_names(loss_class, names):
  for name in names:
    _STRING_TO_LOSS_DICT[name] = loss_class
    if not name.endswith('_loss'):
      _STRING_TO_LOSS_DICT[name + '_loss'] = loss_class


_register_loss_names(
    pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss,
    ['pairwise_absolute_difference_loss'])
_register_loss_names(
    pairwise_cosine_loss.PairwiseCosineLoss,
    ['pairwise_cosine_loss'])
_register_loss_names(
    pairwise_mse_loss.PairwiseMSELoss,
    ['pairwise_mse_loss'])


def _get_loss(loss: Union[base_loss.CounterfactualLoss, str],
              loss_var_name: str = 'loss') -> base_loss.CounterfactualLoss:
  """Returns a `losses.CounterfactualLoss` instance corresponding to `loss`.

  If `loss` is an instance of `losses.CounterfactualLoss` then it is returned
  directly. If `loss` is a string it must be an accepted loss name. A
  value of `None` is also accepted and simply returns `None`.

  Args:
    loss: loss instance. Can be `None`, a string or an instance of
      `losses.CounterfactualLoss`.
    loss_var_name: Name of the loss variable. This is only used for error
      messaging.

  Returns:
    A `CounterfactualLoss` instance.

  Raises:
    ValueError: If `loss` is an unrecognized string.
    TypeError: If `loss` is not an instance of `losses.CounterfactualLoss` or a
      string.
  """
  if loss is None:
    return None
  if isinstance(loss, base_loss.CounterfactualLoss):
    return loss
  if isinstance(loss, str):
    lower_case_loss = loss.lower()
    if lower_case_loss in _STRING_TO_LOSS_DICT:
      return _STRING_TO_LOSS_DICT[lower_case_loss]()
    raise ValueError(
        f'If {loss_var_name} is a string, it must be a (case-insensitive) '
        'match for one of the following supported values: '
        f'{_STRING_TO_LOSS_DICT.keys()}. given: {loss}')
  raise TypeError(
      f'{loss_var_name} must be either of type CounterfactualLoss or string, '
      f'given: {loss} (type: {type(loss)})')
