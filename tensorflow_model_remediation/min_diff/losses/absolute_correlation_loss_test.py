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

"""Tests for AbsoluteCorrelationLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import absolute_correlation_loss as loss_lib


class AbsoluteCorrelationLossTest(tf.test.TestCase):
  """Tests for Absolute Correlation Loss."""

  def testName(self):
    # Default name.
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    self.assertEqual(loss_fn.name, 'absolute_correlation_loss')

    # Custom name.
    loss_fn = loss_lib.AbsoluteCorrelationLoss(name='custom_loss')
    self.assertEqual(loss_fn.name, 'custom_loss')

  def testEnableSummaryHistogram(self):
    loss = loss_lib.AbsoluteCorrelationLoss()
    self.assertTrue(loss.enable_summary_histogram)
    loss = loss_lib.AbsoluteCorrelationLoss(enable_summary_histogram=False)
    self.assertFalse(loss.enable_summary_histogram)

  def testNoWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(0.823209, loss_value)

  def testNegativeCorrelationNoWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])

    loss_value = loss_fn(membership, predictions)
    self.assertAllClose(0.959671, loss_value)

  def testWithWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.3], [0.1], [0.86], [0.06], [0.75]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0.872562, loss_value)

  def testNegativeCorrelationWithWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [2.5], [1.2], [0.9]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0.967827, loss_value)

  def testAllNegativeWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[-1.0], [-2.0], [-2.5], [-1.2], [-0.9]])

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'sample_weight.*cannot contain any negative weights'):
      loss_value = loss_fn(membership, predictions, sample_weights)
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(loss_value)

  def testSomeNegativeWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[1.0], [2.0], [-2.5], [1.2], [0.9]])

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'sample_weight.*cannot contain any negative weights'):
      loss_value = loss_fn(membership, predictions, sample_weights)
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(loss_value)

  def testAllZeroWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
    membership = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0]])
    predictions = tf.constant([[0.12], [0.7], [0.2], [0.86], [0.32]])
    sample_weights = tf.constant([[0.0], [0.0], [0.0], [0.0], [0.0]])

    loss_value = loss_fn(membership, predictions, sample_weights)
    self.assertAllClose(0, loss_value)

  def testGradients(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
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
    expected_gradients = [[0.5481868], [-1.2328656], [1.1707227], [-0.83559656],
                          [0.34955233]]
    self.assertAllClose(expected_gradients, gradients)

  def testGradientsWithAllZeroWeights(self):
    loss_fn = loss_lib.AbsoluteCorrelationLoss()
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

  def testSerialization(self):
    loss = loss_lib.AbsoluteCorrelationLoss(name='custom_name')
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)

    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)
    self.assertIsInstance(deserialized_loss, loss_lib.AbsoluteCorrelationLoss)
    self.assertEqual(deserialized_loss.name, loss.name)


if __name__ == '__main__':
  tf.test.main()
