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

"""Tests for MinDiffModel class."""

import copy
import os
import tempfile
import mock

import tensorflow as tf

from tensorflow_model_remediation.min_diff import losses
from tensorflow_model_remediation.min_diff.keras import utils
from tensorflow_model_remediation.min_diff.keras.models import min_diff_model


def _loss_fn(x, y, w=None):
  # Arbitrary op that uses all 3 inputs.
  w_sum = 0 if w is None else tf.reduce_sum(w)
  return tf.reduce_sum(x) + tf.reduce_sum(y) + w_sum


class DummyLoss(losses.MinDiffLoss):

  def call(self, x, y, w):
    return _loss_fn(x, y, w)


def _get_sequential():
  return tf.keras.Sequential([tf.keras.layers.Dense(1, activation="softmax")])


class MinDiffModelTest(tf.test.TestCase):

  def setUp(self):
    super(MinDiffModelTest, self).setUp()
    self.x = tf.expand_dims(tf.range(100.0, 150.0), axis=-1)
    self.y1 = tf.constant([[1.0], [0.0]] * 25)
    self.y2 = tf.constant([[1.0], [0.0], [0.0], [0.0], [1.0]] * 10)
    self.w = tf.expand_dims(tf.range(0.01, 1.0, 0.02), axis=-1)
    self.batch_size = 5
    # Original dataset without weights.
    self.original_dataset = tf.data.Dataset.from_tensor_slices(
        (self.x, self.y1)).batch(self.batch_size)
    # Original multi output dataset without weights.
    self.original_multi_dataset = tf.data.Dataset.from_tensor_slices((self.x, {
        "y1": self.y1,
        "y2": self.y2
    })).batch(self.batch_size)
    # Original dataset with weights.
    self.original_weighted_dataset = tf.data.Dataset.from_tensor_slices(
        (self.x, self.y1, self.w)).batch(self.batch_size)

    self.min_diff_x = tf.expand_dims(tf.range(200.0, 250.0), axis=-1)
    self.min_diff_mem = tf.constant([[1.0]] * 25 + [[0.0]] * 25)
    self.min_diff_w = tf.expand_dims(tf.range(0.99, 0.0, -0.02), axis=-1)
    self.min_diff_data = (self.min_diff_x, self.min_diff_mem)
    self.min_diff_weighted_data = (self.min_diff_x, self.min_diff_mem,
                                   self.min_diff_w)

    # Dataset with min_diff packed in.
    self.min_diff_dataset = tf.data.Dataset.from_tensor_slices(
        (utils.MinDiffPackedInputs(
            original_inputs=self.x,
            min_diff_data=self.min_diff_data), self.y1)).batch(self.batch_size)
    # Multi output dataset with min_diff packed in.
    self.min_diff_multi_dataset = tf.data.Dataset.from_tensor_slices(
        (utils.MinDiffPackedInputs(
            original_inputs=self.x, min_diff_data=self.min_diff_data), {
                "y1": self.y1,
                "y2": self.y2
            })).batch(self.batch_size)
    # Dataset with weights and min_diff packed in.
    self.min_diff_weighted_dataset = tf.data.Dataset.from_tensor_slices(
        (utils.MinDiffPackedInputs(
            original_inputs=self.x, min_diff_data=self.min_diff_weighted_data),
         self.y1, self.w)).batch(self.batch_size)

  def testIsModel(self):
    model = min_diff_model.MinDiffModel(tf.keras.Sequential(), DummyLoss())
    self.assertIsInstance(model, tf.keras.Model)

  def testSettingLoss(self):
    model = min_diff_model.MinDiffModel(tf.keras.Sequential(), "mmd_loss")
    self.assertIsInstance(model._loss, losses.MMDLoss)

  def testSettingPredictionsTransform(self):
    val = 3  # Arbitrary value.
    # Defaults to the identity.
    model = min_diff_model.MinDiffModel(tf.keras.Sequential(), DummyLoss())
    self.assertEqual(val, model.predictions_transform(val))

    # Correctly sets a provided function.
    model = min_diff_model.MinDiffModel(
        tf.keras.Sequential(),
        DummyLoss(),
        predictions_transform=lambda x: x * 2)
    self.assertEqual(val * 2, model.predictions_transform(val))

    # Raises an error if the parameter passed in is not callable.
    bad_fn = "<not callable>"
    with self.assertRaisesRegex(
        ValueError,
        "predictions_transform.*must be callable.*{}".format(bad_fn)):
      _ = min_diff_model.MinDiffModel(
          tf.keras.Sequential(), DummyLoss(), predictions_transform=bad_fn)

  def testGetDataFns(self):
    original_model = tf.keras.Sequential()
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    x = "fake_x"
    min_diff_data = "fake_min_diff_data"
    packed_inputs = utils.MinDiffPackedInputs(
        original_inputs=x, min_diff_data=min_diff_data)

    self.assertEqual(model.unpack_original_inputs(x), x)
    self.assertEqual(model.unpack_original_inputs(packed_inputs), x)

    self.assertEqual(model.unpack_min_diff_data(packed_inputs), min_diff_data)
    self.assertIsNone(model.unpack_min_diff_data(min_diff_data))

  def testComputeMinDiffLossPassesInputsThrough(self):
    original_model = tf.keras.Sequential()
    # Mock original_model's call function.
    pred_mock = tf.constant([1, 2, 3])  # Arbitrary Tensor values.
    original_model.call = mock.MagicMock(return_value=pred_mock)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())
    model._loss = mock.MagicMock()

    # Training unset.
    _ = model.compute_min_diff_loss(self.min_diff_data)
    original_model.call.assert_called_once_with(
        self.min_diff_x, training=None, mask=None)
    model._loss.assert_called_once_with(
        predictions=pred_mock, membership=self.min_diff_mem, sample_weight=None)

    # Training set to True.
    original_model.call.reset_mock()
    model._loss.reset_mock()
    _ = model.compute_min_diff_loss(
        self.min_diff_weighted_data, training=True, mask="mask")
    original_model.call.assert_called_once_with(
        self.min_diff_x, training=True, mask="mask")
    model._loss.assert_called_once_with(
        predictions=pred_mock,
        membership=self.min_diff_mem,
        sample_weight=self.min_diff_w)

  def testComputeMinDiffLoss(self):
    original_model = tf.keras.Sequential()
    predictions = tf.expand_dims(tf.range(1.0, 51), axis=-1)
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    # Assert correct inference call and calculated loss.
    loss = model.compute_min_diff_loss(self.min_diff_data, training=True)
    original_model.call.assert_called_once_with(
        self.min_diff_x, training=True, mask=None)
    self.assertAllClose(loss, _loss_fn(predictions, self.min_diff_mem))

  def testComputeMinDiffLossWithWeights(self):
    original_model = tf.keras.Sequential()
    predictions = tf.expand_dims(tf.range(1.0, 51), axis=-1)
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    # Assert correct inference call and calculated loss.
    loss = model.compute_min_diff_loss(
        self.min_diff_weighted_data, training=True)
    original_model.call.assert_called_once_with(
        self.min_diff_x, training=True, mask=None)
    self.assertAllClose(
        loss, _loss_fn(predictions, self.min_diff_mem, self.min_diff_w))

  def testCallInputsPassedThrough(self):
    original_model = tf.keras.Sequential()
    pred_mock = "fake_preds"
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=pred_mock)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())
    model.compute_min_diff_loss = mock.MagicMock(return_value=0)

    x = "fake_x"
    min_diff_data = "fake_min_diff_data"
    packed_inputs = utils.MinDiffPackedInputs(
        original_inputs=x, min_diff_data=min_diff_data)

    # Training and mask unset.
    _ = model(packed_inputs)
    original_model.call.assert_called_once_with(x, training=None, mask=None)
    model.compute_min_diff_loss.assert_called_once_with(
        min_diff_data, training=None)

    # Training set to True, mask set to False.
    original_model.call.reset_mock()
    model.compute_min_diff_loss.reset_mock()
    model._clear_losses()
    _ = model(packed_inputs, training=True, mask=False)
    original_model.call.assert_called_once_with(x, training=True, mask=False)
    model.compute_min_diff_loss.assert_called_once_with(
        min_diff_data, training=True)

  def testCallWithMinDiffData(self):
    original_model = tf.keras.Sequential()
    predictions = tf.expand_dims(tf.range(1.0, 51), axis=-1)
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())
    loss_val = 3.4  # Arbitrary value.
    model.compute_min_diff_loss = mock.MagicMock(return_value=loss_val)

    packed_inputs = utils.MinDiffPackedInputs(
        original_inputs=self.x, min_diff_data=self.min_diff_data)

    # Assert correct calls are made, correct loss is added, and correct
    # predictions returned.
    preds = model(packed_inputs, training=True, mask=False)
    original_model.call.assert_called_once_with(
        self.x, training=True, mask=False)
    model.compute_min_diff_loss.assert_called_once_with(
        self.min_diff_data, training=True)
    self.assertAllClose(model.losses, [loss_val])
    self.assertAllClose(preds, predictions)

  def testCallWithoutMinDiffData(self):
    original_model = tf.keras.Sequential()
    predictions = tf.expand_dims(tf.range(1.0, 6), axis=-1)
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())
    model.compute_min_diff_loss = mock.MagicMock()

    # Assert correct calls are made, correct loss is added, and correct
    # predictions returned.
    preds = model(self.x, training=False, mask=True)
    original_model.call.assert_called_once_with(
        self.x, training=False, mask=True)
    model.compute_min_diff_loss.assert_not_called()
    self.assertEmpty(model.losses)
    self.assertAllClose(preds, predictions)

  def testEvalOutputs(self):
    original_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, activation="softmax")])
    model = min_diff_model.MinDiffModel(original_model, losses.MMDLoss("gauss"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    # Evaluate with min_diff_data.
    output_metrics = model.test_step(iter(self.min_diff_dataset).get_next())
    self.assertSetEqual(
        set(output_metrics.keys()), set(["loss", "acc", "min_diff_loss"]))

    # Evaluate without min_diff_data.
    output_metrics = model.test_step(iter(self.original_dataset).get_next())
    self.assertSetEqual(set(output_metrics.keys()), set(["loss", "acc"]))

  def testTrainingWithoutMinDiffDataRaisesError(self):
    original_model = tf.keras.Sequential()
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    with self.assertRaisesRegex(ValueError,
                                "must contain MinDiffData during training"):
      _ = model(self.x, training=True)

  def testCustomModelWithoutMaskInCallSignature(self):

    mock_call = mock.MagicMock(return_value=tf.constant(1.0))

    class CustomModel1(tf.keras.Model):

      def call(self, inputs, training=None):
        # Use mock to track call.
        return mock_call(inputs, training=training)

    original_model = CustomModel1()

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    # Assert different call signature doesn't break and still receives the right
    # value.
    _ = model.compute_min_diff_loss(self.min_diff_data, training=True)
    mock_call.assert_called_once_with(self.min_diff_x, training=True)

    # Assert different call signature doesn't break and still receives the right
    # value.
    mock_call.reset_mock()
    _ = model(self.x, training=False)
    mock_call.assert_called_once_with(self.x, training=False)

  def testCustomModelWithoutTrainingInCallSignature(self):

    mock_call = mock.MagicMock(return_value=tf.constant(1.0))

    class CustomModel1(tf.keras.Model):

      def call(self, inputs, mask=None):
        # Use mock to track call.
        return mock_call(inputs, mask=mask)

    original_model = CustomModel1()

    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    # Assert different call signature doesn't break and still receives the right
    # value.
    _ = model.compute_min_diff_loss(
        self.min_diff_data, training=True, mask=False)
    mock_call.assert_called_once_with(self.min_diff_x, mask=False)

    # Assert different call signature doesn't break and still receives the right
    # value.
    mock_call.reset_mock()
    _ = model(self.x, training=False, mask=False)
    mock_call.assert_called_once_with(self.x, mask=False)

  def testOverwritingUnpackingFunctions(self):
    original_model = tf.keras.Sequential()

    class CustomMinDiffModel(min_diff_model.MinDiffModel):

      def unpack_original_inputs(self, inputs):
        return inputs + "custom_original_inputs"

      def unpack_min_diff_data(self, inputs):
        return inputs + "min_diff_data"

    model = CustomMinDiffModel(original_model, DummyLoss())

    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value="fake_preds")
    model.compute_min_diff_loss = mock.MagicMock()

    x = "x_"

    # Assert correct calls are made, correct loss is added, and correct
    # predictions returned.
    _ = model(x)
    original_model.call.assert_called_once_with(
        "x_custom_original_inputs", training=None, mask=None)
    model.compute_min_diff_loss.assert_called_once_with(
        "x_min_diff_data", training=None)

  # TODO: consider testing actual output. This is not currently
  # done because the disproportionate amount of complexity this would add.
  def testSingleOutputModel(self):
    original_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, activation="softmax")])
    model = min_diff_model.MinDiffModel(original_model, losses.MMDLoss("gauss"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    history = model.fit(self.min_diff_dataset)
    self.assertSetEqual(
        set(history.history.keys()), set(["loss", "acc", "min_diff_loss"]))

    # Evaluate with min_diff_data.
    model.evaluate(self.min_diff_dataset)

    # Evaluate and run inference without min_diff_data.
    model.evaluate(self.original_dataset)
    model.predict(self.original_dataset)

  def testBadPredictionsTransformReturnValueRaisesError(self):
    original_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, activation="softmax")])
    bad_value = "<not a tensor>"
    model = min_diff_model.MinDiffModel(
        original_model,
        DummyLoss(),
        predictions_transform=lambda preds: bad_value)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    with self.assertRaisesRegex(
        ValueError, "predictions.*must be a Tensor.*{}.*\n.*the provided"
        ".*predictions_transform.*does not return a Tensor".format(bad_value)):
      _ = model.fit(self.min_diff_dataset)

  def testMultiOutputModel(self):
    # Create multi output Functional model.
    inputs = tf.keras.Input(1)
    output_1 = tf.keras.layers.Dense(1, activation="softmax")(inputs)
    output_2 = tf.keras.layers.Dense(1, activation="softmax")(inputs)
    original_model = tf.keras.Model(
        inputs=inputs, outputs={
            "y1": output_1,
            "y2": output_2,
        })
    model = min_diff_model.MinDiffModel(
        original_model,
        DummyLoss(),
        predictions_transform=lambda preds: preds["y1"])

    model.compile(
        loss={
            "y1": "binary_crossentropy",
            "y2": "binary_crossentropy",
        },
        optimizer="adam",
        metrics=["acc"])

    history = model.fit(self.min_diff_multi_dataset)
    self.assertSetEqual(
        set(history.history.keys()),
        set(["loss", "y1_loss", "y1_acc", "y2_loss", "y2_acc",
             "min_diff_loss"]))

    # Evaluate with min_diff_data.
    model.evaluate(self.min_diff_multi_dataset)

    # Evaluate and run inference without min_diff_data.
    model.evaluate(self.original_multi_dataset)
    model.predict(self.original_multi_dataset)

  def testMissingPredictionTransformRaisesErrorForMultiOutput(self):
    # Create multi output Functional model.
    inputs = tf.keras.Input(1)
    output_1 = tf.keras.layers.Dense(1, activation="softmax")(inputs)
    output_2 = tf.keras.layers.Dense(1, activation="softmax")(inputs)
    original_model = tf.keras.Model(
        inputs=inputs, outputs={
            "y1": output_1,
            "y2": output_2
        })
    # Not setting a predictions_transform for a multi output original model will
    # fail during training.
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    model.compile(
        loss={
            "y1": "binary_crossentropy",
            "y2": "binary_crossentropy"
        },
        optimizer="adam",
        metrics=["acc"])

    with self.assertRaisesRegex(
        ValueError, "predictions.*must be a Tensor.*y1.*y2.*\n.*original_model"
        ".*does not return a Tensor.*pass in.*predictions_transform"):
      _ = model.fit(self.min_diff_multi_dataset)

  def testCustomModelWithSequential(self):

    class CustomSequential(tf.keras.Sequential):

      def __init__(self, *args, **kwargs):
        super(CustomSequential, self).__init__(*args, **kwargs)
        # Variable will be incremented in the custom train_step.
        self.train_step_cnt = 0

      def train_step(self, data):
        self.train_step_cnt += 1
        return super(CustomSequential, self).train_step(data)

    original_model = CustomSequential(
        [tf.keras.layers.Dense(1, activation="softmax")])

    class CustomMinDiffModel(min_diff_model.MinDiffModel, CustomSequential):
      pass

    model = CustomMinDiffModel(original_model, DummyLoss())

    # Compile with `run_eagerly=True` so that we can expect the number of times
    # `train_step` is called to be the same as the number of batches passed in.
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["acc"],
        run_eagerly=True)

    history = model.fit(self.min_diff_dataset)
    self.assertSetEqual(
        set(history.history.keys()), set(["loss", "acc", "min_diff_loss"]))
    # There are 10 batches so the custom train step should have been called 10
    # times.
    self.assertEqual(model.train_step_cnt, 10)

    # Evaluate with min_diff_data.
    model.evaluate(self.min_diff_dataset)

    # Evaluate and run inference without min_diff_data.
    model.evaluate(self.original_dataset)
    model.predict(self.original_dataset)

  def testWithRegularizationInOriginalModel(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
        tf.keras.layers.ActivityRegularization(0.34)
    ])
    model = min_diff_model.MinDiffModel(
        copy.deepcopy(original_model), DummyLoss())

    # Compile with `run_eagerly=True` so that we can evaluate model.losses.
    compile_args = {
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ["acc"],
        "run_eagerly": True
    }
    original_model.compile(**compile_args)
    model.compile(**compile_args)

    _ = original_model.fit(self.original_dataset)
    _ = model.fit(self.min_diff_dataset)

    # We expect only 1 regularization element in the original model's training.
    self.assertEqual(len(original_model.losses), 1)
    # We expect 2 regularization elements in the min_diff model's training. The
    # first corresponds to the min_diff loss, the second is the original_model's
    # regularization loss.
    self.assertEqual(len(model.losses), 2)

    # The regularization elements should be the same.
    self.assertAllClose(original_model.losses[-1], model.losses[-1])

  def testCustomMinDiffModelWithRegularizationInOriginalModel(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
        tf.keras.layers.ActivityRegularization(0.34)
    ])

    class CustomMinDiffModel(min_diff_model.MinDiffModel, tf.keras.Sequential):
      pass

    model = CustomMinDiffModel(copy.deepcopy(original_model), DummyLoss())

    # Compile with `run_eagerly=True` so that we can evaluate model.losses.
    compile_args = {
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ["acc"],
        "run_eagerly": True
    }
    original_model.compile(**compile_args)
    model.compile(**compile_args)

    _ = original_model.fit(self.original_dataset)
    _ = model.fit(self.min_diff_dataset)

    # We expect only 1 regularization element in the original model's training.
    self.assertEqual(len(original_model.losses), 1)
    # We expect 2 regularization elements in the min_diff model's training. The
    # first corresponds to the min_diff loss, the second is the original_model's
    # regularization loss.
    self.assertEqual(len(model.losses), 2)

    # The regularization elements should be the same.
    self.assertAllClose(original_model.losses[-1], model.losses[-1])

  def testSaveForContinuedTraining(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    model = min_diff_model.MinDiffModel(original_model, losses.MMDLoss())

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = model.fit(self.min_diff_dataset)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      model.save(path)

      loaded_model = tf.keras.models.load_model(path)

    self.assertIsInstance(loaded_model, min_diff_model.MinDiffModel)

    # Run more training on loaded_model.
    loaded_model.fit(self.min_diff_dataset)

  def testSaveForInference(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = model.fit(self.min_diff_dataset)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      model.save_original_model(path)

      loaded_model = tf.keras.models.load_model(path)

    self.assertIsInstance(loaded_model, tf.keras.Sequential)

    # Run inference on loaded_model.
    loaded_model.predict(self.original_dataset)

  def testSaveForInferenceThenRewrap(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = model.fit(self.min_diff_dataset)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      model.save_original_model(path)

      loaded_model = tf.keras.models.load_model(path)

    model = min_diff_model.MinDiffModel(loaded_model, DummyLoss())
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = model.fit(self.min_diff_dataset)

    # Evaluate with min_diff_data.
    model.evaluate(self.min_diff_dataset)

    # Evaluate and run inference without min_diff_data.
    model.evaluate(self.original_dataset)
    model.predict(self.original_dataset)

  def testSaveAndLoadWeights(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    model = min_diff_model.MinDiffModel(original_model, DummyLoss())

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = model.fit(self.min_diff_dataset)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      model.save_weights(path)

      original_model = tf.keras.Sequential([
          tf.keras.layers.Dense(1, activation="softmax"),
      ])
      loaded_model = min_diff_model.MinDiffModel(original_model, DummyLoss())
      loaded_model.compile(
          loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
      loaded_model.load_weights(path)

    _ = model.fit(self.min_diff_dataset)

    # Evaluate with min_diff_data.
    model.evaluate(self.min_diff_dataset)

    # Evaluate and run inference without min_diff_data.
    model.evaluate(self.original_dataset)
    model.predict(self.original_dataset)

  def testSerialization(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    loss_weight = 2.3  # Arbitrary value.
    model_name = "custom_model_name"  # Arbitrary name.
    model = min_diff_model.MinDiffModel(
        original_model,
        losses.MMDLoss(),
        loss_weight=loss_weight,
        name=model_name)

    serialized_model = tf.keras.utils.serialize_keras_object(model)
    deserialized_model = tf.keras.layers.deserialize(
        serialized_model)

    self.assertIsInstance(deserialized_model, min_diff_model.MinDiffModel)
    self.assertIsNone(deserialized_model._predictions_transform)
    self.assertIsInstance(deserialized_model._loss, losses.MMDLoss)
    self.assertEqual(deserialized_model._loss_weight, loss_weight)
    self.assertEqual(deserialized_model.name, model_name)

  def testSerializationWithTransformAndKernel(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    predictions_fn = lambda x: x * 5.1  # Arbitrary operation.

    loss_weight = 2.3  # Arbitrary value.
    model_name = "custom_model_name"  # Arbitrary name.
    model = min_diff_model.MinDiffModel(
        original_model,
        losses.MMDLoss("laplacian"),  # Non-default Kernel.
        loss_weight=loss_weight,
        predictions_transform=predictions_fn,
        name=model_name)

    serialized_model = tf.keras.utils.serialize_keras_object(model)
    deserialized_model = tf.keras.layers.deserialize(
        serialized_model)

    self.assertIsInstance(deserialized_model, min_diff_model.MinDiffModel)
    val = 7  # Arbitrary value.
    self.assertEqual(
        deserialized_model._predictions_transform(val), predictions_fn(val))
    self.assertIsInstance(deserialized_model._loss, losses.MMDLoss)
    self.assertIsInstance(deserialized_model._loss.predictions_kernel,
                          losses.LaplacianKernel)
    self.assertEqual(deserialized_model._loss_weight, loss_weight)
    self.assertEqual(deserialized_model.name, model_name)

  def testConfig(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    model = min_diff_model.MinDiffModel(original_model, losses.MMDLoss())

    config = model.get_config()

    self.assertSetEqual(
        set(config.keys()),
        set(["original_model", "loss", "loss_weight", "name"]))

    # Test building the model from the config.
    model_from_config = min_diff_model.MinDiffModel.from_config(config)
    self.assertIsInstance(model_from_config, min_diff_model.MinDiffModel)
    self.assertIsInstance(model_from_config.original_model, tf.keras.Sequential)

  def testConfigWithCustomBaseImplementation(self):

    class CustomModel(tf.keras.Model):

      def __init__(self, val, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.val = val

      def get_config(self):
        return {"val": self.val}

    class CustomMinDiffModel(min_diff_model.MinDiffModel, CustomModel):
      pass  # No additional implementation needed.

    original_val = 4  # Arbitrary value passed in.
    original_model = CustomModel(original_val)

    min_diff_model_val = 5  # Different arbitrary value passed in.
    model = CustomMinDiffModel(
        original_model=original_model,
        loss=losses.MMDLoss(),
        val=min_diff_model_val)
    self.assertEqual(model.val, min_diff_model_val)

    config = model.get_config()

    self.assertSetEqual(
        set(config.keys()),
        set(["original_model", "loss", "loss_weight", "name", "val"]))
    self.assertEqual(config["val"], model.val)
    self.assertEqual(config["original_model"].val, original_model.val)

    # Test building the model from the config.
    model_from_config = CustomMinDiffModel.from_config(config)
    self.assertIsInstance(model_from_config, CustomMinDiffModel)
    self.assertIsInstance(model_from_config.original_model, CustomModel)
    self.assertEqual(model_from_config.val, min_diff_model_val)
    self.assertEqual(model_from_config.original_model.val, original_val)

  def testGetConfigErrorFromOriginalModel(self):

    class CustomModel(tf.keras.Model):
      pass  # No need to add any other implementation for this test.

    original_model = CustomModel()

    model = min_diff_model.MinDiffModel(original_model, losses.MMDLoss())

    with self.assertRaisesRegex(
        NotImplementedError, "MinDiffModel cannot create a config.*"
        "original_model.*not implemented get_config.*or has an error"):
      _ = model.get_config()


if __name__ == "__main__":
  tf.test.main()
