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

"""Tests for CounterfactualLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.counterfactual.losses import base_loss


@tf.keras.utils.register_keras_serializable()
class CustomLoss(base_loss.CounterfactualLoss):

  def __init__(self, name=None):
    super(CustomLoss, self).__init__(name=name)

  def call(self):
    pass  # Dummy Placeholder. Will not be called unless subclassed.


class CounterfactualLossLossTest(tf.test.TestCase):

  def testAbstract(self):

    with self.assertRaisesRegex(
        TypeError,
        '.*instantiate abstract class CounterfactualLoss with abstract.*'):
      base_loss.CounterfactualLoss()

    class CustomLoss1(base_loss.CounterfactualLoss):
      pass

    with self.assertRaisesRegex(TypeError,
                                'instantiate abstract class CustomLoss1'):
      CustomLoss1()

    class CustomLoss2(base_loss.CounterfactualLoss):

      def call(self):
        pass

    CustomLoss2()

  def testLossName(self):
    # Default name.
    loss = CustomLoss()
    self.assertEqual(loss.name, 'custom_loss')

    # Custom name.
    loss = CustomLoss(name='custom_name')
    self.assertEqual(loss.name, 'custom_name')

    # Private class
    class _CustomLoss(base_loss.CounterfactualLoss):

      def call(self):
        pass

    # Private name name.
    loss = _CustomLoss()
    self.assertEqual(loss.name, 'private__custom_loss')

  def testReduction(self):
    # CounterfactualLoss should set the reduction to NONE.
    loss = CustomLoss()
    self.assertEqual(loss.reduction, tf.keras.losses.Reduction.NONE)

  def testGetAndFromConfig(self):
    loss = CustomLoss()
    config = loss.get_config()
    self.assertDictEqual(config, {'name': loss.name})

    loss_from_config = CustomLoss.from_config(config)

    self.assertIsInstance(loss_from_config, CustomLoss)

if __name__ == '__main__':
  tf.test.main()
