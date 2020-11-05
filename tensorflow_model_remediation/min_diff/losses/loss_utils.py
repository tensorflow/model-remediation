# coding=utf-8
# Copyright 2020 Google LLC.
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

# Lint as: python3
"""Utils for min diff losses."""

from typing import Text, Union

from tensorflow_model_remediation.min_diff.losses import absolute_correlation_loss as abscorrloss
from tensorflow_model_remediation.min_diff.losses import base_loss
from tensorflow_model_remediation.min_diff.losses import mmd_loss

import six

_LOSSES_DICT = {
    'abscorr': abscorrloss.AbsoluteCorrelationLoss,
    'abscorrloss': abscorrloss.AbsoluteCorrelationLoss,
    'absolute_correlation': abscorrloss.AbsoluteCorrelationLoss,
    'absolute_correlation_loss': abscorrloss.AbsoluteCorrelationLoss,
    'mmd': mmd_loss.MMDLoss,
    'mmd_loss': mmd_loss.MMDLoss,
}


def _get_loss(loss: Union[base_loss.MinDiffLoss, Text],
              loss_var_name: Text = 'loss') -> base_loss.MinDiffLoss:
  """Returns a `losses.MinDiffLoss` instance corresponding to `loss`.

  If `loss` is an instance of `losses.MinDiffLoss` then it is returned
  directly. If `loss` is a string it must be an accepted loss name. A
  value of `None` is also accepted and simply returns `None`.

  Args:
    loss: loss instance. Can be `None`, a string or an instance of
      `losses.MinDiffLoss`.
    loss_var_name: Name of the loss variable. This is only used for error
      messaging.

  Returns:
    A `MinDiffLoss` instance.

  Raises:
    ValueError: If `loss` is an unrecognized string.
    TypeError: If `loss` is not an instance of `losses.MinDiffLoss` or a string.
  """
  if loss is None:
    return None
  if isinstance(loss, base_loss.MinDiffLoss):
    return loss
  if isinstance(loss, six.string_types):
    lower_case_loss = loss.lower()
    if lower_case_loss in _LOSSES_DICT:
      return _LOSSES_DICT[lower_case_loss]()
    raise ValueError('If {} is a string, it must be a (case-insensitive) '
                     'match for one of the following supported values: {}. '
                     'given: {}'.format(loss_var_name, _LOSSES_DICT.keys(),
                                        loss))
  raise TypeError('{} must be either of type MinDiffLoss or string, given: '
                  '{} (type: {})'.format(loss_var_name, loss, type(loss)))
