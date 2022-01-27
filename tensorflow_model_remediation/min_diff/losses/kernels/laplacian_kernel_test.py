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

"""Test laplacian_kernel module."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel


class LaplacianKernelTest(tf.test.TestCase):

  def testLaplacianKernel(self):
    # Under certain conditions, the kernel output resembles the identity
    # (within the tolerance of the checks).
    # Specifically, this is true when the minimum distance, d, between any two
    # elements in the input is greater than D for D = -ln(tol) * kernel_length
    # where:
    #   - tol: assertion tolerance (these tests ues 1e-6)
    #   - d: minimum{norm(x_i - y_i)
    #                for x_i = x[i, :], y_j = y[j, :] for all i neq j}
    #   - kernel_length: argument to kernel initialization
    #
    # In the test below, we use max(kernel_length) == 0.1 which gives us
    # max(D)~=1.38. Given this bound, we sometines use d=2 (meaning that each
    # element is at least 2 greater or smaller than every other element in the
    # input). When we do this, we expect the output to be close to the identity.

    for kernel_length in [0.05, 0.075, 0.1]:
      laplacian_kernel_fn = laplacian_kernel.LaplacianKernel(kernel_length)

      kernel_val = laplacian_kernel_fn(tf.constant([[2.0], [0.0]]))
      self.assertAllClose(kernel_val, tf.eye(2))

      kernel_val = laplacian_kernel_fn(tf.constant([[1.0], [8.0], [3.0]]))
      self.assertAllClose(kernel_val, tf.eye(3))

      kernel_val = laplacian_kernel_fn(
          tf.constant([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]))
      self.assertAllClose(kernel_val, tf.eye(3))

      # Vector where all elements are equal should result in matrix of all 1s.
      for val in [0.2, 1.0, 2.79]:  # Arbitrary values.
        kernel_val = laplacian_kernel_fn(tf.constant([[val, val], [val, val]]))
        # self.assertAllClose(kernel_val, tf.eye(3))
        self.assertAllClose(kernel_val, [[1.0, 1.0], [1.0, 1.0]])

      # If the two tensors are equal, it should be the same as a single tensor.
      # Note that we pick d < 1 so that the result is not the identity.
      tensor = tf.constant([[1.0], [1.01]])
      single_kernel_val = laplacian_kernel_fn(tensor)
      # Assert that the result is nontrivial (not the identity).
      self.assertNotAllClose(tf.eye(2), single_kernel_val)

      double_kernel_val = laplacian_kernel_fn(tensor, tensor)
      self.assertAllClose(single_kernel_val, double_kernel_val)
      self.assertIsNot(single_kernel_val, double_kernel_val)

      # If the delta is the same, then the result should be the same.
      # Note that we pick d < 1 so that the result is not the identity.
      tensor1 = tf.constant([[1.0], [7.0]])
      tensor2 = tf.constant([[1.0], [5.3]])
      delta = tf.constant([[0.0], [0.2]])
      kernel_val1 = laplacian_kernel_fn(tensor1, tensor1 + delta)
      # Assert that the result is nontrivial (not the identity).
      self.assertNotAllClose(tf.eye(2), kernel_val1)

      kernel_val2 = laplacian_kernel_fn(tensor2, tensor2 + delta)
      self.assertAllClose(kernel_val1, kernel_val2)

  def testGetAndFromConfig(self):
    kernel_length = 5  # Arbitrary value.
    kernel = laplacian_kernel.LaplacianKernel(kernel_length)

    kernel_from_config = laplacian_kernel.LaplacianKernel.from_config(
        kernel.get_config())
    self.assertIsInstance(kernel_from_config, laplacian_kernel.LaplacianKernel)
    self.assertEqual(kernel_from_config.kernel_length, kernel_length)

  def testSerialization(self):
    kernel_length = 5  # Arbitrary value.
    kernel = laplacian_kernel.LaplacianKernel(kernel_length)

    serialized_kernel = tf.keras.utils.serialize_keras_object(kernel)

    deserialized_kernel = tf.keras.utils.deserialize_keras_object(
        serialized_kernel)
    self.assertIsInstance(deserialized_kernel, laplacian_kernel.LaplacianKernel)
    self.assertEqual(deserialized_kernel.kernel_length, kernel_length)


if __name__ == '__main__':
  tf.test.main()
