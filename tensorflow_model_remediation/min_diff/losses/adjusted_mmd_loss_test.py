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

"""Tests for AdjustedMMDLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import adjusted_mmd_loss as adjusted_mmd_lib


class AdjustedMMDLossTest(tf.test.TestCase):
  """Tests for Adjusted MMD Loss."""

  def testName(self):
    # Default name.
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss()
    self.assertEqual(loss_fn.name, "adjusted_mmd_loss")

    # Custom name.
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(name="custom_loss")
    self.assertEqual(loss_fn.name, "custom_loss")

  def testEnableSummaryHistogram(self):
    loss = adjusted_mmd_lib.AdjustedMMDLoss()
    self.assertTrue(loss.enable_summary_histogram)
    loss = adjusted_mmd_lib.AdjustedMMDLoss(enable_summary_histogram=False)
    self.assertFalse(loss.enable_summary_histogram)

  def testGaussianKernelNoWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(8.627613, loss_value)

  def testGaussianKernelNegativeCorrelationNoWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(23.2143, loss_value, atol=1e-4)

  def testGaussianKernelWithWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(9.935723, loss_value)

  def testGaussianKernelNegativeCorrelationWithWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(25.3270, loss_value, atol=1e-4)

  def testGaussianKernelSomeZeroWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(9.935723, loss_value)

  def testGaussianKernelAllZeroWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[0.0], [0.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0, loss_value)

  def testLaplacianKernelNoWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(
        "laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(6.4339886, loss_value)

  def testLaplacianKernelNegativeCorrelationNoWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(
        "laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(7.7721, loss_value, atol=1e-4)

  def testLaplacianKernelWithWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(
        "laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(7.163468, loss_value)

  def testLaplacianKernelNegativeCorrelationWithWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(
        "laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(8.1804, loss_value, atol=1e-4)

  def testLaplacianKernelSomeZeroWeights(self):
    loss_fn = adjusted_mmd_lib.AdjustedMMDLoss(
        "laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9], [0.0],
                                  [0.0]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(7.163468, loss_value)


if __name__ == "__main__":
  tf.test.main()
