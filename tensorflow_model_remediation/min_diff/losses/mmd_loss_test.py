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

"""Test mmd_loss module."""

from model_remediation.min_diff.losses import gauss_kernel
from model_remediation.min_diff.losses import mmd_loss as loss_lib
import tensorflow as tf


class MMDLossTest(tf.test.TestCase):
  """Tests for MMD Loss."""

  def testName(self):
    # Default name.
    loss_fn = loss_lib.MMDLoss()
    self.assertEqual(loss_fn.name, "mmd_loss")

    # Custom name.
    loss_fn = loss_lib.MMDLoss(name="custom_loss")
    self.assertEqual(loss_fn.name, "custom_loss")

  def testGaussKernelNoWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.083264, loss_value)

  def testGaussKernelNegativeCorrelationNoWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.464054, loss_value)

  def testGaussKernelWithWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussKernelNegativeCorrelationWithWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.481506, loss_value)

  def testGaussKernelSomeZeroWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussKernelAllZeroWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[0.0], [0.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0, loss_value)

  def testGaussKernelSomeNegativeWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[1.0], [2.0], [-2.5], [1.2], [0.9]]

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "sample_weight.*cannot contain any negative weights"):
      loss_value = loss_fn(membership, predictions, sample_weights)
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(loss_value)

  def testGaussKernelAllNegativeWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[-1.0], [-2.0], [-2.5], [-1.2], [-0.9]]

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "sample_weight.*cannot contain any negative weights"):
      loss_value = loss_fn(membership, predictions, sample_weights)
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(loss_value)

  def testGaussKernelMultiDimTensor(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.0, 0.5], [1.0, 0.0], [0.8, 0.0], [0.0, 0.8],
                               [0.75, 0.8], [0.2, 0.4], [0.5, 0.1]])
    sample_weights = [[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0.262881, loss_value)

  def testLaplaceKernelNoWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace")
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(0.970659, loss_value)

  def testLaplaceKernelNegativeCorrelationNoWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace")
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.119343, loss_value)

  def testLaplaceKernelWithWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace")
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testLaplaceKernelNegativeCorrelationWithWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace")
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.159840, loss_value)

  def testLaplaceKernelSomeZeroWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace")
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9], [0.0],
                                  [0.0]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testRaisesExpectedErrors(self):
    kernel = gauss_kernel.GaussKernel()
    loss_lib.MMDLoss(kernel)
    bad_kernel = lambda x: kernel(x)  # pylint:disable=unnecessary-lambda
    with self.assertRaisesRegex(
        TypeError,
        "predictions_kernel.*must be.*MinDiffKernel.*string.*4.*int"):
      loss_lib.MMDLoss(4)
    with self.assertRaisesRegex(
        TypeError,
        "predictions_kernel.*must be.*MinDiffKernel.*string.*lambda"):
      loss_lib.MMDLoss(bad_kernel)


if __name__ == "__main__":
  tf.test.main()
