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

from tensorflow_model_remediation.min_diff.losses import base_kernel
from tensorflow_model_remediation.min_diff.losses import gauss_kernel
from tensorflow_model_remediation.min_diff.losses import laplace_kernel
import six

_KERNELS_DICT = {
    'gauss': gauss_kernel.GaussianKernel,
    'gausskernel': gauss_kernel.GaussianKernel,
    'laplace': laplace_kernel.LaplacianKernel,
    'laplacekernel': laplace_kernel.LaplacianKernel,
}


def _get_kernel(kernel: Union[base_kernel.MinDiffKernel, Text],
                kernel_var_name: Text = 'kernel'):
  """Returns a `losses.MinDiffKernel` instance corresponding to `kernel`.

  If `kernel` is an instance of `losses.MinDiffKernel` then it is returned
  directly. If `kernel` is a string it must be an accepted kernel name. A
  value of `None` is also accepted and simply returns `None`.

  Args:
    kernel: kernel instance. Can be `None`, a string or an instance of
      `losses.MinDiffKernel`.
    kernel_var_name: Name of the kernel variable. This is only used for error
      messaging.

  Returns:
    Returns a MinDiffKernel instance.
  """
  if kernel is None:
    return None
  if isinstance(kernel, base_kernel.MinDiffKernel):
    return kernel
  if isinstance(kernel, six.string_types):
    lower_case_kernel = kernel.lower()
    if lower_case_kernel in _KERNELS_DICT:
      return _KERNELS_DICT[lower_case_kernel]()
    raise ValueError('If {} is a string, it must be a (case-insensitive) '
                     'match for one of the following supported values: {}. '
                     'given: {}'.format(kernel_var_name, _KERNELS_DICT.keys(),
                                        kernel))
  raise TypeError('{} must be either of type MinDiffKernel or string, given: '
                  '{} (type: {})'.format(kernel_var_name, kernel, type(kernel)))
