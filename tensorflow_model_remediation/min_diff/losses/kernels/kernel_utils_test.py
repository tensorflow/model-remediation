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

"""Test kernel_utils module."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import kernel_utils as utils
from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel


class GetMinDiffKernelTest(tf.test.TestCase):

  def testAcceptsNone(self):
    kernel = utils._get_kernel(None)
    self.assertIsNone(kernel)

  def testForGaussianKernel(self):
    kernel = utils._get_kernel('gauss')
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    kernel = utils._get_kernel('GauSs')  # Strangely capitalized.
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    kernel = utils._get_kernel('gauss_kernel')
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    kernel = utils._get_kernel('gaussian')
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    kernel = utils._get_kernel('gaussian_kernel')
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    kernel_length = 3
    kernel = utils._get_kernel(gaussian_kernel.GaussianKernel(kernel_length))
    self.assertIsInstance(kernel, gaussian_kernel.GaussianKernel)
    self.assertEqual(kernel.kernel_length, kernel_length)

  def testForLaplacianKernel(self):
    kernel = utils._get_kernel('laplace')
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    kernel = utils._get_kernel('laplace_Kernel')
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    kernel = utils._get_kernel('laplacian')
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    kernel = utils._get_kernel('laplacian_kernel')
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    kernel = utils._get_kernel(laplacian_kernel.LaplacianKernel())
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    kernel_length = 3
    kernel = utils._get_kernel(laplacian_kernel.LaplacianKernel(kernel_length))
    self.assertIsInstance(kernel, laplacian_kernel.LaplacianKernel)
    self.assertEqual(kernel.kernel_length, kernel_length)

  def testForCustomKernel(self):

    class CustomKernel(base_kernel.MinDiffKernel):

      def call(self, x, y):
        pass

    kernel = CustomKernel()
    kernel_output = utils._get_kernel(kernel)
    self.assertIs(kernel_output, kernel)

  def testGetKernelRaisesErrors(self):
    with self.assertRaisesRegex(
        TypeError, 'custom_name.*must be.*MinDiffKernel.*string.*4.*int'):
      utils._get_kernel(4, 'custom_name')

    with self.assertRaisesRegex(
        ValueError, 'custom_name.*must be.*supported values.*bad_name'):
      utils._get_kernel('bad_name', 'custom_name')


if __name__ == '__main__':
  tf.test.main()
