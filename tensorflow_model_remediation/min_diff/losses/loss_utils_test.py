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

"""Tests for loss_utils functions."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import absolute_correlation_loss as abscorrloss
from tensorflow_model_remediation.min_diff.losses import adjusted_mmd_loss
from tensorflow_model_remediation.min_diff.losses import base_loss
from tensorflow_model_remediation.min_diff.losses import loss_utils as utils
from tensorflow_model_remediation.min_diff.losses import mmd_loss


class GetMinDiffLossTest(tf.test.TestCase):

  def testAcceptsNone(self):
    loss = utils._get_loss(None)
    self.assertIsNone(loss)

  def testForAbsoluteCorrelationLoss(self):
    loss = utils._get_loss('abs_corr')
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    loss = utils._get_loss('abS_coRr')  # Strangely capitalized.
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    loss = utils._get_loss('abs_corr_loss')  # Other accepted name.
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    loss = utils._get_loss('absolute_correlation')  # Other accepted name.
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    loss = utils._get_loss('absolute_correlation_loss')  # Other accepted name.
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(abscorrloss.AbsoluteCorrelationLoss(loss_name))
    self.assertIsInstance(loss, abscorrloss.AbsoluteCorrelationLoss)
    self.assertEqual(loss.name, loss_name)

  def testForMMDLoss(self):
    loss = utils._get_loss('mmd')
    self.assertIsInstance(loss, mmd_loss.MMDLoss)
    loss = utils._get_loss('mmd_loss')
    self.assertIsInstance(loss, mmd_loss.MMDLoss)
    loss = utils._get_loss(mmd_loss.MMDLoss())
    self.assertIsInstance(loss, mmd_loss.MMDLoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(mmd_loss.MMDLoss(name=loss_name))
    self.assertIsInstance(loss, mmd_loss.MMDLoss)
    self.assertEqual(loss.name, loss_name)

  def testForAdjustedMMDLoss(self):
    loss = utils._get_loss('adjusted_mmd')
    self.assertIsInstance(loss, adjusted_mmd_loss.AdjustedMMDLoss)
    loss = utils._get_loss('adjusted_mmd_loss')
    self.assertIsInstance(loss, adjusted_mmd_loss.AdjustedMMDLoss)
    loss = utils._get_loss(mmd_loss.MMDLoss())
    self.assertIsInstance(loss, mmd_loss.MMDLoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(adjusted_mmd_loss.AdjustedMMDLoss(name=loss_name))
    self.assertIsInstance(loss, adjusted_mmd_loss.AdjustedMMDLoss)
    self.assertEqual(loss.name, loss_name)

  def testForCustomLoss(self):

    class CustomLoss(base_loss.MinDiffLoss):

      def call(self, x, y):
        pass

    loss = CustomLoss()
    loss_output = utils._get_loss(loss)
    self.assertIs(loss_output, loss)

  def testGetLossRaisesErrors(self):
    with self.assertRaisesRegex(
        TypeError, 'custom_name.*must be.*MinDiffLoss.*string.*4.*int'):
      utils._get_loss(4, 'custom_name')

    with self.assertRaisesRegex(
        ValueError, 'custom_name.*must be.*supported values.*bad_name'):
      utils._get_loss('bad_name', 'custom_name')


if __name__ == '__main__':
  tf.test.main()
