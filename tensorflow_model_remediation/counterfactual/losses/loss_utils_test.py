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

from tensorflow_model_remediation.counterfactual.losses import base_loss
from tensorflow_model_remediation.counterfactual.losses import loss_utils as utils
from tensorflow_model_remediation.counterfactual.losses import pairwise_absolute_difference_loss
from tensorflow_model_remediation.counterfactual.losses import pairwise_cosine_loss
from tensorflow_model_remediation.counterfactual.losses import pairwise_mse_loss


class GetCounterfactualLossTest(tf.test.TestCase):

  def testAcceptsNone(self):
    loss = utils._get_loss(None)
    self.assertIsNone(loss)

  def testForCounterfactualLoss(self):

    class CustomLoss(base_loss.CounterfactualLoss):

      def call(self, x, y):
        pass

    loss = CustomLoss()
    loss_output = utils._get_loss(loss)
    self.assertIs(loss_output, loss)

  def testForPairwiseMSELoss(self):
    loss = utils._get_loss('pairwise_mse_loss')
    self.assertIsInstance(loss, pairwise_mse_loss.PairwiseMSELoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(pairwise_mse_loss.PairwiseMSELoss(name=loss_name))
    self.assertIsInstance(loss, pairwise_mse_loss.PairwiseMSELoss)
    self.assertEqual(loss.name, loss_name)

  def testForPairwiseCosineLoss(self):
    loss = utils._get_loss('pairwise_cosine_loss')
    self.assertIsInstance(loss, pairwise_cosine_loss.PairwiseCosineLoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(
        pairwise_cosine_loss.PairwiseCosineLoss(name=loss_name))
    self.assertIsInstance(loss, pairwise_cosine_loss.PairwiseCosineLoss)
    self.assertEqual(loss.name, loss_name)

  def testForPairwiseAbsoluteDifferenceLoss(self):
    loss = utils._get_loss('pairwise_absolute_difference_loss')
    self.assertIsInstance(
        loss, pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss)
    loss_name = 'custom_name'
    loss = utils._get_loss(
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss(
            name=loss_name))
    self.assertIsInstance(
        loss,
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss)
    self.assertEqual(loss.name, loss_name)

  def testGetLossRaisesErrors(self):
    with self.assertRaisesRegex(
        TypeError, 'custom_name.*must be.*CounterfactualLoss.*string.*4.*int'):
      utils._get_loss(4, 'custom_name')

    with self.assertRaisesRegex(
        ValueError, 'custom_name.*must be.*supported values.*bad_name'):
      utils._get_loss('bad_name', 'custom_name')


if __name__ == '__main__':
  tf.test.main()
