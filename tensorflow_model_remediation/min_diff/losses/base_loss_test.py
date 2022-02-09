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

"""Tests for MinDiffLoss class."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.losses import base_loss
from tensorflow_model_remediation.min_diff.losses.kernels import base_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import gaussian_kernel
from tensorflow_model_remediation.min_diff.losses.kernels import laplacian_kernel


@tf.keras.utils.register_keras_serializable()
class CustomLoss(base_loss.MinDiffLoss):

  def __init__(self,
               membership_transform=None,
               predictions_transform=None,
               membership_kernel=None,
               predictions_kernel=None,
               name=None):
    super(CustomLoss, self).__init__(
        membership_transform=membership_transform,
        predictions_transform=predictions_transform,
        membership_kernel=membership_kernel,
        predictions_kernel=predictions_kernel,
        name=name)

  def call(self):
    pass  # Dummy Placeholder. Will not be called unless subclassed.


class CustomKernel(base_kernel.MinDiffKernel):

  def __init__(self, num):
    super(CustomKernel, self).__init__(tile_input=False)
    self.num = num

  def call(self, x, y):
    del y  # Unused.
    return x + self.num  # Arbitrary op.


class MinDiffLossTest(tf.test.TestCase):

  def testAbstract(self):

    with self.assertRaisesRegex(TypeError,
                                'instantiate abstract class MinDiffLoss'):
      base_loss.MinDiffLoss()

    class CustomLoss1(base_loss.MinDiffLoss):
      pass

    with self.assertRaisesRegex(TypeError,
                                'instantiate abstract class CustomLoss1'):
      CustomLoss1()

    class CustomLoss2(base_loss.MinDiffLoss):

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
    class _CustomLoss(base_loss.MinDiffLoss):

      def call(self):
        pass

    # Private name name.
    loss = _CustomLoss()
    self.assertEqual(loss.name, 'private__custom_loss')

  def testReduction(self):
    # MinDiffLoss should set the reduction to NONE.
    loss = CustomLoss()
    self.assertEqual(loss.reduction, tf.keras.losses.Reduction.NONE)

  def testTransformAttributesDefaultToNone(self):
    loss = CustomLoss()
    self.assertIsNone(loss.membership_transform)
    self.assertIsNone(loss.predictions_transform)

  def testTransformAttributes(self):
    transform_1 = lambda x: x + 2  # Arbitrary transform.
    transform_2 = lambda x: x + 3  # Different arbitrary transform.
    val = 7  # Arbitrary value.

    loss = CustomLoss(
        membership_transform=transform_1, predictions_transform=transform_2)
    self.assertEqual(loss.membership_transform(val), transform_1(val))
    self.assertEqual(loss.predictions_transform(val), transform_2(val))

    loss = CustomLoss(predictions_transform=transform_1)
    self.assertEqual(loss.predictions_transform(val), transform_1(val))
    self.assertIsNone(loss.membership_transform)

    loss = CustomLoss(membership_transform=transform_2)
    self.assertEqual(loss.membership_transform(val), transform_2(val))
    self.assertIsNone(loss.predictions_transform)

  def testTransformInputRaisesErrors(self):
    with self.assertRaisesRegex(ValueError, 'should be a callable instance'):
      _ = CustomLoss(membership_transform='not callable')
    with self.assertRaisesRegex(ValueError, 'should be a callable instance'):
      _ = CustomLoss(predictions_transform='not callable')

  def testKernelAttributesDefaultsToNone(self):
    loss = CustomLoss()
    self.assertIsNone(loss.membership_kernel)
    self.assertIsNone(loss.predictions_kernel)

  def testKernelAttributes(self):
    loss = CustomLoss(membership_kernel='gauss', predictions_kernel='laplace')
    self.assertIsInstance(loss.membership_kernel,
                          gaussian_kernel.GaussianKernel)
    self.assertIsInstance(loss.predictions_kernel,
                          laplacian_kernel.LaplacianKernel)

    kernel = gaussian_kernel.GaussianKernel()
    loss = CustomLoss(predictions_kernel=kernel)
    self.assertIs(loss.predictions_kernel, kernel)
    self.assertIsNone(loss.membership_kernel)

    kernel = laplacian_kernel.LaplacianKernel()
    loss = CustomLoss(membership_kernel=kernel)
    self.assertIs(loss.membership_kernel, kernel)
    self.assertIsNone(loss.predictions_kernel)

  def testApplyKernels(self):
    kernel1 = CustomKernel(3)  # Arbitrary input.
    kernel2 = CustomKernel(7)  # Arbitrary input.
    val1 = 1  # Arbitrary value.
    val2 = 2  # Arbitrary value.
    loss = CustomLoss(membership_kernel=kernel1, predictions_kernel=kernel2)
    membership_kernel, predictions_kernel = loss._apply_kernels(val1, val2)
    self.assertEqual(membership_kernel, kernel1(val1))
    self.assertEqual(predictions_kernel, kernel2(val2))

    loss = CustomLoss(predictions_kernel=kernel2)
    membership, predictions_kernel = loss._apply_kernels(val1, val2)
    self.assertEqual(membership, val1)
    self.assertEqual(predictions_kernel, kernel2(val2))

    loss = CustomLoss(membership_kernel=kernel1)
    membership_kernel, predictions = loss._apply_kernels(val1, val2)
    self.assertEqual(membership_kernel, kernel1(val1))
    self.assertEqual(predictions, val2)

  def testPreprocessInputs(self):
    transform_1 = lambda x: x + 2  # Arbitrary transform.
    transform_2 = lambda x: x + 3  # Different arbitrary transform.
    val1 = tf.constant([1])  # Arbitrary value.
    val2 = tf.constant([2])  # Arbitrary value.

    loss = CustomLoss(
        membership_transform=transform_1, predictions_transform=transform_2)
    membership, predictions, weights = loss._preprocess_inputs(val1, val2, None)
    self.assertAllClose(membership, transform_1(val1))
    self.assertAllClose(predictions, transform_2(val2))
    self.assertAllClose(weights, [[1]])

    loss = CustomLoss(predictions_transform=transform_2)
    membership, predictions, weights = loss._preprocess_inputs(val1, val2, None)
    self.assertAllClose(membership, val1)
    self.assertAllClose(predictions, transform_2(val2))
    self.assertAllClose(weights, [[1]])

    loss = CustomLoss(membership_transform=transform_1)
    membership, predictions, weights = loss._preprocess_inputs(val1, val2, None)
    self.assertAllClose(membership, transform_1(val1))
    self.assertAllClose(predictions, val2)
    self.assertAllClose(weights, [[1]])

    # Weights normalized
    val1 = tf.constant([1, 1])  # Arbitrary value.
    val2 = tf.constant([2, 2])  # Arbitrary value.

    _, _, weights = loss._preprocess_inputs(val1, val2, None)
    self.assertAllClose(weights, [[0.5], [0.5]])

    _, _, weights = loss._preprocess_inputs(val1, val2, tf.constant([[1], [3]]))
    self.assertAllClose(weights, [[0.25], [0.75]])

  def testPreprocessInputsRaisesErrors(self):
    loss = CustomLoss()
    # Weights normalized
    val1 = tf.constant([1, 1])  # Arbitrary value.
    val2 = tf.constant([2, 2])  # Arbitrary value.

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'sample_weight.*cannot contain any negative weights'):
      op = loss._preprocess_inputs(val1, val2, tf.constant([[-1], [3]]))
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          sess.run(op)

  def testGetAndFromConfig(self):
    loss = CustomLoss()
    config = loss.get_config()
    self.assertDictEqual(config, {'name': loss.name})

    loss_from_config = CustomLoss.from_config(config)

    self.assertIsInstance(loss_from_config, CustomLoss)

  def testSerialization(self):
    loss = CustomLoss()
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, CustomLoss)
    self.assertIsNone(deserialized_loss.membership_transform)
    self.assertIsNone(deserialized_loss.predictions_transform)
    self.assertIsNone(deserialized_loss.membership_kernel)
    self.assertIsNone(deserialized_loss.predictions_kernel)
    self.assertEqual(deserialized_loss.name, loss.name)

  def testSerializationWithTransformsAndKernels(self):
    membership_fn = lambda x: x * 2.3  # Arbitrary operation.
    predictions_fn = lambda x: x * 5.1  # Arbitrary operation.

    loss = CustomLoss(
        membership_transform=membership_fn,
        membership_kernel='gaussian',
        predictions_transform=predictions_fn,
        predictions_kernel='laplacian')
    serialized_loss = tf.keras.utils.serialize_keras_object(loss)
    deserialized_loss = tf.keras.utils.deserialize_keras_object(serialized_loss)

    self.assertIsInstance(deserialized_loss, CustomLoss)
    val = 7  # Arbitrary value.
    self.assertEqual(
        deserialized_loss.membership_transform(val), membership_fn(val))
    self.assertEqual(
        deserialized_loss.predictions_transform(val), predictions_fn(val))
    self.assertIsInstance(deserialized_loss.membership_kernel,
                          gaussian_kernel.GaussianKernel)
    self.assertIsInstance(deserialized_loss.predictions_kernel,
                          laplacian_kernel.LaplacianKernel)
    self.assertEqual(deserialized_loss.name, loss.name)


if __name__ == '__main__':
  tf.test.main()
