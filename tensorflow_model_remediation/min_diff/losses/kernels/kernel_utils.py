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

"""Utils for MinDiff kernels."""

from typing import Union

from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel

_STRING_TO_KERNEL_DICT = {}


def _register_kernel_names(kernel_class, names):
  for name in names:
    _STRING_TO_KERNEL_DICT[name] = kernel_class
    if not name.endswith('_kernel'):
      _STRING_TO_KERNEL_DICT[name + '_kernel'] = kernel_class


_register_kernel_names(gaussian_kernel.GaussianKernel, ['gauss', 'gaussian'])
_register_kernel_names(laplacian_kernel.LaplacianKernel,
                       ['laplace', 'laplacian'])


def _get_kernel(kernel: Union[base_kernel.MinDiffKernel, str],
                kernel_var_name: str = 'kernel'):
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
  if isinstance(kernel, str):
    lower_case_kernel = kernel.lower()
    if lower_case_kernel in _STRING_TO_KERNEL_DICT:
      return _STRING_TO_KERNEL_DICT[lower_case_kernel]()
    raise ValueError('If {} is a string, it must be a (case-insensitive) '
                     'match for one of the following supported values: {}. '
                     'given: {}'.format(kernel_var_name,
                                        _STRING_TO_KERNEL_DICT.keys(), kernel))
  raise TypeError('{} must be either of type MinDiffKernel or string, given: '
                  '{} (type: {})'.format(kernel_var_name, kernel, type(kernel)))
