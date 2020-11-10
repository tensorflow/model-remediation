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

"""Test gaussian_kernel module."""

from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
import tensorflow as tf


class GaussianKernelTest(tf.test.TestCase):

  def testGaussianKernel(self):
    # Under certain conditions, the kernel output resembles the identity
    # (within the tolerance of the checks).
    # Specifically, this is true when the minimum distance, d, between any two
    # elements in the input is greater than D = -ln(tol) * kernel_length**2
    # where:
    #   - tol: assertion tolerance (these tests ues 1e-6)
    #   - d: minimum{norm(x_i - y_i)
    #                for x_i = x[i, :], y_j = y[j, :] for all i neq j}
    #   - kernel_length: argument to kernel initialization
    #
    # In the test below, we use max(kernel_length) == 0.1 which gives us
    # max(D)~=0.138. Given this bound, we sometines use d=0.2 (meaning that each
    # element is at least 0.2 greater or smaller than every other element in the
    # input). When we do this, we expect the output to be close to the identity.
    for kernel_length in [0.05, 0.075, 0.1]:
      gaussian_kernel_fn = gaussian_kernel.GaussianKernel(kernel_length)
      kernel_val = gaussian_kernel_fn(tf.constant([[1.0], [0.0]]))
      self.assertAllClose(kernel_val, tf.eye(2))

      kernel_val = gaussian_kernel_fn(tf.constant([[1.0], [2.0], [3.0]]))
      self.assertAllClose(kernel_val, tf.eye(3))

      kernel_val = gaussian_kernel_fn(
          tf.constant([[0.1, 0.3], [0.5, 0.7], [0.9, 1.1]]))
      self.assertAllClose(kernel_val, tf.eye(3))

      # Vector where all elements are equal should result in matrix of all 1s.
      for val in [0.2, 1.0, 2.79]:  # Arbitrary values.
        kernel_val = gaussian_kernel_fn(tf.constant([[val, val], [val, val]]))
        self.assertAllClose(kernel_val, tf.ones((2, 2)))

      # If tensors are equal, it should be the same as single tensor.
      # Note that we pick d < 0.1 so that the result is not the identity.
      tensor = tf.constant([[1.0], [1.01]])
      single_kernel_val = gaussian_kernel_fn(tensor)
      # Assert that the result is nontrivial (not the identity).
      self.assertNotAllClose(tf.eye(2), single_kernel_val)

      double_kernel_val = gaussian_kernel_fn(tensor, tensor)
      self.assertAllClose(single_kernel_val, double_kernel_val)

      # If the delta is the same, then the result should be the same.
      # Note that we pick d < 0.1 so that the result is not the identity.
      tensor1 = tf.constant([[1.0], [2.003]])
      tensor2 = tf.constant([[1.0], [5.503]])
      delta = tf.constant([[0.0], [0.01]])
      kernel_val1 = gaussian_kernel_fn(tensor1, tensor1 + delta)
      # Assert that the result is nontrivial (not the identity).
      self.assertNotAllClose(tf.eye(2), kernel_val1)

      kernel_val2 = gaussian_kernel_fn(tensor2, tensor2 + delta)
      self.assertAllClose(kernel_val1, kernel_val2)


if __name__ == '__main__':
  tf.test.main()
