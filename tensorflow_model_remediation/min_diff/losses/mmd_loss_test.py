# coding=utf-8
# Copyright 2021 Google LLC.
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

"""Tests for MMDLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import mmd_loss as loss_lib
from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel


class MMDLossTest(tf.test.TestCase):
  """Tests for MMD Loss."""

  def testName(self):
    # Default name.
    loss_fn = loss_lib.MMDLoss()
    self.assertEqual(loss_fn.name, "mmd_loss")

    # Custom name.
    loss_fn = loss_lib.MMDLoss(name="custom_loss")
    self.assertEqual(loss_fn.name, "custom_loss")

  def testGaussianKernelNoWeights(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.083264, loss_value)

  def testGaussianKernelNegativeCorrelationNoWeights(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.464054, loss_value)

  def testGaussianKernelWithWeights(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussianKernelNegativeCorrelationWithWeights(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.481506, loss_value)

  def testGaussianKernelSomeZeroWeights(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussianKernelAllZeroWeights(self):
    loss_fn = loss_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[0.0], [0.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0, loss_value)

  def testGaussianKernelSomeNegativeWeights(self):
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

  def testGaussianKernelAllNegativeWeights(self):
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

  def testGaussianKernelMultiDimTensor(self):
    loss_fn = loss_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.0, 0.5], [1.0, 0.0], [0.8, 0.0], [0.0, 0.8],
                               [0.75, 0.8], [0.2, 0.4], [0.5, 0.1]])
    sample_weights = [[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0.262881, loss_value)

  def testLaplacianKernelNoWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(0.970659, loss_value)

  def testLaplacianKernelNegativeCorrelationNoWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.119343, loss_value)

  def testLaplacianKernelWithWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testLaplacianKernelNegativeCorrelationWithWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.159840, loss_value)

  def testLaplacianKernelSomeZeroWeights(self):
    loss_fn = loss_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9], [0.0],
                                  [0.0]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testRaisesExpectedErrors(self):
    kernel = gaussian_kernel.GaussianKernel()
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

  def testSerialization(self):
    loss = loss_lib.MMDLoss()
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, loss_lib.MMDLoss)
    self.assertIsNone(deserialized_loss.predictions_transform)
    self.assertIsInstance(deserialized_loss.predictions_kernel,
                          gaussian_kernel.GaussianKernel)
    self.assertEqual(deserialized_loss.name, loss.name)

  def testSerializationWithTransformAndKernel(self):
    predictions_fn = lambda x: x * 5.1  # Arbitrary operation.

    loss = loss_lib.MMDLoss(
        predictions_transform=predictions_fn, kernel="laplacian")
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, loss_lib.MMDLoss)
    val = 7  # Arbitrary value.
    self.assertEqual(
        deserialized_loss.predictions_transform(val), predictions_fn(val))
    self.assertIsInstance(deserialized_loss.predictions_kernel,
                          laplacian_kernel.LaplacianKernel)
    self.assertEqual(deserialized_loss.name, loss.name)


if __name__ == "__main__":
  tf.test.main()
