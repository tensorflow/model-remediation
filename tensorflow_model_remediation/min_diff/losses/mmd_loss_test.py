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

"""Tests for MMDLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import mmd_loss as mmd_lib
from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel


class MMDLossTest(tf.test.TestCase):
  """Tests for MMD Loss."""

  def testName(self):
    # Default name.
    loss_fn = mmd_lib.MMDLoss()
    self.assertEqual(loss_fn.name, "mmd_loss")

    # Custom name.
    loss_fn = mmd_lib.MMDLoss(name="custom_loss")
    self.assertEqual(loss_fn.name, "custom_loss")

  def testEnableSummaryHistogram(self):
    loss = mmd_lib.MMDLoss()
    self.assertTrue(loss.enable_summary_histogram)
    loss = mmd_lib.MMDLoss(enable_summary_histogram=False)
    self.assertFalse(loss.enable_summary_histogram)

  def testGaussianKernelNoWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.083264, loss_value)

  def testGaussianKernelNegativeCorrelationNoWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.464054, loss_value)

  def testGaussianKernelWithWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussianKernelNegativeCorrelationWithWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.481506, loss_value)

  def testGaussianKernelSomeZeroWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = [[1.0], [2.0], [2.5], [1.2], [0.9], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.293555, loss_value)

  def testGaussianKernelAllZeroWeights(self):
    loss_fn = mmd_lib.MMDLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = [[0.0], [0.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0, loss_value)

  def testGaussianKernelSomeNegativeWeights(self):
    loss_fn = mmd_lib.MMDLoss()
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
    loss_fn = mmd_lib.MMDLoss()
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
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.0, 0.5], [1.0, 0.0], [0.8, 0.0], [0.0, 0.8],
                               [0.75, 0.8], [0.2, 0.4], [0.5, 0.1]])
    sample_weights = [[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0.262881, loss_value)

  def testGaussianGradients(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    variables = tf.constant([[0.1], [0.3], [0.5], [0.7], [0.9]])

    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    with tf.GradientTape() as tape:
      tape.watch(variables)
      predictions = variables * 3  # arbitrary linear operation.
      loss_value = loss_fn(membership, predictions, sample_weights)

    gradients = tape.gradient(loss_value, variables)
    # Assert that gradient computations are non trivial and do not change based
    # on loss implementation.
    expected_gradients = [[-0.85786223], [-1.8886726], [1.1220325], [0.5379708],
                          [-0.02113436]]
    self.assertAllClose(expected_gradients, gradients)

  def testGaussianGradientsAllZeroWeights(self):
    loss_fn = mmd_lib.MMDLoss(predictions_transform=tf.sigmoid)
    variables = tf.constant([[0.1], [0.3], [0.5], [0.7], [0.9]])

    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    sample_weights = tf.constant([[0.0], [0.0], [0.0], [0.0], [0.0]])

    with tf.GradientTape() as tape:
      tape.watch(variables)
      predictions = variables * 3  # arbitrary linear operation.
      loss_value = loss_fn(membership, predictions, sample_weights)

    gradients = tape.gradient(loss_value, variables)
    # Gradients should all be 0 for weights that are all 0.
    expected_gradients = [[0.0], [0.0], [0.0], [0.0], [0.0]]
    self.assertAllClose(expected_gradients, gradients)

  def testLaplacianKernelNoWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(0.970659, loss_value)

  def testLaplacianKernelNegativeCorrelationNoWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(1.119343, loss_value)

  def testLaplacianKernelWithWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testLaplacianKernelNegativeCorrelationWithWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.159840, loss_value)

  def testLaplacianKernelSomeZeroWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75], [0.2],
                               [0.5]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9], [0.0],
                                  [0.0]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(1.122741, loss_value)

  def testLaplacianGradients(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    variables = tf.constant([[0.1], [0.3], [0.5], [0.7], [0.9]])

    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    with tf.GradientTape() as tape:
      tape.watch(variables)
      predictions = variables * 3  # arbitrary linear operation.
      loss_value = loss_fn(membership, predictions, sample_weights)

    gradients = tape.gradient(loss_value, variables)
    # Assert that gradient computations are non trivial and do not change based
    # on loss implementation.
    expected_gradients = [[-0.40014654], [-0.7467264], [0.39164954],
                          [0.10974818], [0.08942482]]
    self.assertAllClose(expected_gradients, gradients)

  def testLaplacianGradientsAllZeroWeights(self):
    loss_fn = mmd_lib.MMDLoss("laplace", predictions_transform=tf.sigmoid)
    variables = tf.constant([[0.1], [0.3], [0.5], [0.7], [0.9]])

    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    sample_weights = tf.constant([[0.0], [0.0], [0.0], [0.0], [0.0]])

    with tf.GradientTape() as tape:
      tape.watch(variables)
      predictions = variables * 3  # arbitrary linear operation.
      loss_value = loss_fn(membership, predictions, sample_weights)

    gradients = tape.gradient(loss_value, variables)
    # Gradients should all be 0 for weights that are all 0.
    expected_gradients = [[0.0], [0.0], [0.0], [0.0], [0.0]]
    self.assertAllClose(expected_gradients, gradients)

  def testRaisesExpectedErrors(self):
    kernel = gaussian_kernel.GaussianKernel()
    mmd_lib.MMDLoss(kernel)
    bad_kernel = lambda x: kernel(x)  # pylint:disable=unnecessary-lambda
    with self.assertRaisesRegex(
        TypeError,
        "predictions_kernel.*must be.*MinDiffKernel.*string.*4.*int"):
      mmd_lib.MMDLoss(4)
    with self.assertRaisesRegex(
        TypeError,
        "predictions_kernel.*must be.*MinDiffKernel.*string.*lambda"):
      mmd_lib.MMDLoss(bad_kernel)

  def testSerialization(self):
    loss = mmd_lib.MMDLoss()
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, mmd_lib.MMDLoss)
    self.assertIsNone(deserialized_loss.predictions_transform)
    self.assertIsInstance(deserialized_loss.predictions_kernel,
                          gaussian_kernel.GaussianKernel)
    self.assertEqual(deserialized_loss.name, loss.name)

  def testSerializationWithTransformAndKernel(self):
    predictions_fn = lambda x: x * 5.1  # Arbitrary operation.

    loss = mmd_lib.MMDLoss(
        predictions_transform=predictions_fn, kernel="laplacian")
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, mmd_lib.MMDLoss)
    val = 7  # Arbitrary value.
    self.assertEqual(
        deserialized_loss.predictions_transform(val), predictions_fn(val))
    self.assertIsInstance(deserialized_loss.predictions_kernel,
                          laplacian_kernel.LaplacianKernel)
    self.assertEqual(deserialized_loss.name, loss.name)

if __name__ == "__main__":
  tf.test.main()
