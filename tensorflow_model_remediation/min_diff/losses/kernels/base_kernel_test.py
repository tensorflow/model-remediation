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

"""Test base_kernel module."""

from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel
import tensorflow as tf


class MinDiffKernelTest(tf.test.TestCase):

  def testAbstract(self):

    with self.assertRaisesRegex(TypeError,
                                'instantiate abstract class MinDiffKernel'):
      base_kernel.MinDiffKernel()

    class CustomKernel(base_kernel.MinDiffKernel):
      pass

    with self.assertRaisesRegex(TypeError,
                                'instantiate abstract class CustomKernel'):
      CustomKernel()


if __name__ == '__main__':
  tf.test.main()
