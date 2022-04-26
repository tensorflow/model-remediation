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

"""Test pairwise_cosine_loss module."""

import tensorflow as tf

from tensorflow_model_remediation.counterfactual.losses import pairwise_cosine_loss


class PairwiseCosineLossTest(tf.test.TestCase):

  def testAbsoluteMatch(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[4., 8.], [12., 8.], [1., 3.]])

    loss = loss_obj(original, original)

    self.assertAlmostEqual(self.evaluate(loss), -1.0, 2)

  def testUnweighted(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9], [2, -5], [-2, 6]], dtype=tf.float32)
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual)

    self.assertAlmostEqual(self.evaluate(loss), -0.511, 3)

  def testScalarWeighted(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9], [2, -5], [-2, 6]], dtype=tf.float32)
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual, sample_weight=2.3)

    self.assertAlmostEqual(self.evaluate(loss), -1.175, 3)

  def testLossWith2DShapeForSampleWeight(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9, 2], [-5, -2, 6]], dtype=tf.float32)
    counterfactual = tf.constant([[4, 8, 12], [8, 1, 3]], dtype=tf.float32)
    sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    self.assertEqual(sample_weight.shape, [2, 1])
    self.assertAlmostEqual(self.evaluate(loss), 0.156, 2)

  def testLossWith1DShapeForSampleWeight(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9, 2], [-5, -2, 6]], dtype=tf.float32)
    counterfactual = tf.constant([[4, 8, 12], [8, 1, 3]], dtype=tf.float32)
    sample_weight = tf.constant([1.2, 3.4])

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    self.assertEqual(sample_weight.shape, [2])
    self.assertAlmostEqual(self.evaluate(loss), 0.156, 2)

  def testSparseTensors(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.sparse.from_dense([[1., 0., 2., 0], [3., 0., 0., 4.]])
    counterfactual = tf.sparse.from_dense([[1., 1., 2., 0.], [7., 0., 0., 4.]])
    sample_weight = tf.constant([1.2, 0.5])

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    self.assertAllClose(self.evaluate(loss), -0.777, 1e-3)

  def testZeroWeighted(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9], [2, -5], [-2, 6]], dtype=tf.float32)
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual, sample_weight=0)

    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def testMismatchedShapeForOriginalAndCounterfactual(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[4, 8]])

    with self.assertRaisesRegex(
        (ValueError, tf.errors.InvalidArgumentError),
        (r'Incompatible shape \[3, 2\] vs \[1, 2\]|Dimensions must be equal.')):
      loss_obj(original, counterfactual)

  def testNoneInput(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()

    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError),
                                (r'Argument `original` must not be None.')):
      loss_obj(None, None)

  def testEmptyInput(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()

    self.assertAlmostEqual(
        self.evaluate(loss_obj(tf.constant([]), tf.constant([]))), 0)

  def testMismatchedSampleWeightShape(self):
    loss_obj = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[1, 9], [2, -5], [-2, 6]])
    sample_weight = tf.constant([1, 2, 3, 4])

    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError), (
        r'Incompatible `sample_weight` shape \[4\]. Must be scalar or 1D tensor'
        r' of shape \[batch_size\] i.e, \[3\].')):
      loss_obj(original, counterfactual, sample_weight=sample_weight)

  def test_serialisable(self):
    loss = pairwise_cosine_loss.PairwiseCosineLoss()
    original = tf.constant([[1.0, 9.0], [2.0, -5.0], [-2.0, 6.0]])
    counterfactual = tf.constant([[4.0, 8.0], [12.0, 8.0], [1.0, 3.0]])

    serialized = tf.keras.utils.serialize_keras_object(loss)
    deserialized = tf.keras.utils.deserialize_keras_object(serialized)
    original_output = loss(original, counterfactual)
    deserialized_output = deserialized(original, counterfactual)

    self.assertDictEqual(loss.get_config(), deserialized.get_config())
    self.assertAllClose(original_output, deserialized_output)


if __name__ == '__main__':
  tf.test.main()
