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

"""Test base_kernel module."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel


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

  def testGetAndFromConfig(self):

    class CustomKernel(base_kernel.MinDiffKernel):

      def __init__(self, arg, **kwargs):
        super(CustomKernel, self).__init__(**kwargs)
        self.arg = arg

      def call(self, x, y):
        pass  # Unused in this test.

      def get_config(self):
        config = super(CustomKernel, self).get_config()
        config.update({'arg': self.arg})
        return config

    val = 5
    kernel = CustomKernel(val)

    kernel_from_config = CustomKernel.from_config(kernel.get_config())
    self.assertIsInstance(kernel_from_config, CustomKernel)
    self.assertEqual(kernel_from_config.arg, val)
    self.assertTrue(kernel_from_config.tile_input)

    kernel = CustomKernel(val, tile_input=False)

    kernel_from_config = CustomKernel.from_config(kernel.get_config())
    self.assertIsInstance(kernel_from_config, CustomKernel)
    self.assertEqual(kernel_from_config.arg, val)
    self.assertFalse(kernel_from_config.tile_input)


if __name__ == '__main__':
  tf.test.main()
