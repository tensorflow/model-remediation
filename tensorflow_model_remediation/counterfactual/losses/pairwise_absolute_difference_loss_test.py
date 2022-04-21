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

"""Test paired_absolute_difference_loss module."""

import tensorflow as tf

from tensorflow_model_remediation.counterfactual.losses import pairwise_absolute_difference_loss


class PairwiseAbsoluteDifferenceLossTest(tf.test.TestCase):

  def testAbsoluteMatch(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[4., 8.], [12., 8.], [1., 3.]])

    loss = loss_obj(original, original)

    self.assertAlmostEqual(self.evaluate(loss), 0.0, 2)

  def testUnweighted(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual)

    self.assertAlmostEqual(self.evaluate(loss), 5.5, 3)

  def testScalarWeighted(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual, sample_weight=2.3)

    self.assertAlmostEqual(self.evaluate(loss), 12.65, 3)

  def testLossWith2DShapeForSampleWeight(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9, 2], [-5, -2, 6]])
    counterfactual = tf.constant([[4, 8, 12], [8, 1, 3]], dtype=tf.float32)
    sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    self.assertEqual(sample_weight.shape, [2, 1])
    self.assertAlmostEqual(self.evaluate(loss), 13.57, 2)

  def testLossWith1DShapeForSampleWeight(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9, 2], [-5, -2, 6]])
    counterfactual = tf.constant([[4, 8, 12], [8, 1, 3]], dtype=tf.float32)
    sample_weight = tf.constant([1.2, 3.4])

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    self.assertEqual(sample_weight.shape, [2])
    self.assertAlmostEqual(self.evaluate(loss), 13.57, 2)

  def testRaggedTensors(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.ragged.constant([[1., 1., 9.], [2., 5.]])
    counterfactual = tf.ragged.constant([[4., 1., 8.], [12., 3.]])
    sample_weight = tf.constant([1.2, 0.5])

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    # abs_diff = [(|4 - 1|+ |8 - 9|) / 3, (|12 - 2| + |3 - 5|) / 2]
    # abs_diff = [1.33, 6]
    # weighted_abs_diff = [1.33 * 1.2, 6 * 0.5] = [1.5991, 3.0]
    # reduced_weighted_abs_diff = (1.5991 + 3.0) / 2 = 2.29
    self.assertAllClose(self.evaluate(loss), 2.3, 1e-3)

  def testSparseTensors(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.sparse.from_dense([[1., 0., 2., 0], [3., 0., 0., 4.]])
    counterfactual = tf.sparse.from_dense([[1., 1., 2., 0.], [7., 0., 0., 4.]])
    sample_weight = tf.constant([1.2, 0.5])

    loss = loss_obj(original, counterfactual, sample_weight=sample_weight)

    # abs_diff = [(|1 - 1|+ |0 - 1| + |2-2| + |0-1|) / 4,
    #             (|3 - 7| + |0 - 0| + |0 - 0| + |4-4|) / 4]
    # abs_diff = [0.25, 1]
    # weighted_abs_diff = [0.25 * 1.2, 1 * 0.5] = [0.3, 0.5]
    # reduced_weighted_abs_diff = (0.3 + 0.5) / 2 = 0.4
    self.assertAllClose(self.evaluate(loss), 0.4, 1e-3)

  def testZeroWeighted(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[4, 8], [12, 8], [1, 3]], dtype=tf.float32)

    loss = loss_obj(original, counterfactual, sample_weight=0)

    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def testMismatchedShapeForOriginalAndCounterfactual(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[4, 8]])

    with self.assertRaisesRegex(
        (ValueError, tf.errors.InvalidArgumentError),
        (r'Incompatible shape \[3, 2\] vs \[1, 2\]|Dimensions must be equal.')):
      loss_obj(original, counterfactual)

  def testNoneInput(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())

    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError),
                                (r'Argument `original` must not be None.')):
      loss_obj(None, None)

  def testEmptyInput(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())

    self.assertAlmostEqual(
        self.evaluate(loss_obj(tf.constant([]), tf.constant([]))), 0)

  def testMismatchedSampleWeightShape(self):
    loss_obj = (
        pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss())
    original = tf.constant([[1, 9], [2, -5], [-2, 6]])
    counterfactual = tf.constant([[1, 9], [2, -5], [-2, 6]])
    sample_weight = tf.constant([1, 2, 3, 4])

    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError), (
        r'Incompatible `sample_weight` shape \[4\]. Must be scalar or 1D tensor'
        r' of shape \[batch_size\] i.e, \[3\].')):
      loss_obj(original, counterfactual, sample_weight=sample_weight)

  def test_serialisable(self):
    loss = pairwise_absolute_difference_loss.PairwiseAbsoluteDifferenceLoss()
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
