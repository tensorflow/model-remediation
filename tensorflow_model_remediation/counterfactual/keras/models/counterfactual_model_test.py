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

"""Tests for counterfactual_model."""

import copy
import os
import tempfile

from unittest import mock

import tensorflow as tf
from tensorflow_model_remediation.counterfactual import losses
from tensorflow_model_remediation.counterfactual.keras import utils
from tensorflow_model_remediation.counterfactual.keras.models import counterfactual_model


def get_original_model():

  class SumLayer(tf.keras.layers.Layer):

    def build(self, _):
      self.w = self.add_weight("w", ())

    def call(self, inputs):
      return tf.keras.backend.sum(inputs, axis=1, keepdims=True) + self.w * 0

  model = tf.keras.Sequential([SumLayer(input_shape=(3,))])
  return model


class BatchCounterCallback(tf.keras.callbacks.Callback):

  def __init__(self):
    self.batch_begin_count = 0
    self.batch_end_count = 0

  def on_batch_begin(self, *args, **kwargs):
    self.batch_begin_count += 1

  def on_batch_end(self, *args, **kwargs):
    self.batch_end_count += 1


class CounterfactualModelTest(tf.test.TestCase):

  def setUp(self):
    super(CounterfactualModelTest, self).setUp()
    self.original_x = tf.constant([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0],
                                   [-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]])
    self.y = tf.constant([0.0, 0.0, 0.0, 0.0])
    self.w = tf.constant([5.0, 6.0, 7.0, 8.0])
    self.counterfactual_x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    self.cf_w = tf.constant([1.0, 2.0, 3.0, 4.0])

    self.original_input = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.y, self.w))
    self.cf_dataset = tf.data.Dataset.from_tensor_slices(
        (self.original_x, self.counterfactual_x, self.cf_w))
    self.packed_dataset = utils.pack_counterfactual_data(
        self.original_input, self.cf_dataset)

  def testIsModel(self):
    cf_model = counterfactual_model.CounterfactualModel(
        tf.keras.Sequential(), losses.PairwiseMSELoss())
    self.assertIsInstance(cf_model, tf.keras.Model)

  def testSettingLoss(self):
    cf_model = counterfactual_model.CounterfactualModel(tf.keras.Sequential(),
                                                        "pairwise_mse_loss")
    self.assertIsInstance(cf_model._counterfactual_losses,
                          losses.CounterfactualLoss)

  def testUniqueMetricNameHelper(self):
    existing_metrics = [
        tf.keras.metrics.Mean("mean"),
        tf.keras.metrics.MeanSquaredError("mean_1"),
        tf.keras.metrics.Mean("mean_2"),
        tf.keras.metrics.Mean("metric_1"),
        tf.keras.metrics.Mean("unrelated_name"),
    ]
    # Completely new name is unchanged.
    unique_name = counterfactual_model._unique_metric_name(
        "unique", existing_metrics)
    self.assertEqual(unique_name, "unique")
    # Name that is a prefix of others but not included itself is unchanged.
    unique_name = counterfactual_model._unique_metric_name(
        "metric", existing_metrics)
    self.assertEqual(unique_name, "metric")
    # Name that already exists should increment until unique.
    unique_name = counterfactual_model._unique_metric_name(
        "mean", existing_metrics)
    self.assertEqual(unique_name, "mean_3")

  def testUniqueMetricNameInModel(self):

    class CustomModel(tf.keras.Sequential):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric1 = tf.keras.metrics.Mean("counterfactual_loss")
        self._metric2 = tf.keras.metrics.Mean("counterfactual_loss_1")

      @property
      def metrics(self):
        return [self._metric1, self._metric2]

    original_model = CustomModel()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    self.assertEqual(cf_model._counterfactual_loss_metrics.name,
                     "counterfactual_loss_2")

  def testBadLossStructureRaisesError(self):
    # Assert bad structure (nested dict) raises an error.
    with self.assertRaisesRegex(
        ValueError, "`loss` is not a recognized "
        "Counterfactual structure.*unnested.*Given.*"):
      _ = counterfactual_model.CounterfactualModel(tf.keras.Sequential(), {
          "k1": losses.PairwiseMSELoss(),
          "k2": {
              "k": losses.PairwiseMSELoss()
          }
      })
    # Assert bad structure (list) raises an error.
    with self.assertRaisesRegex(
        TypeError, "`loss` is not a recognized "
        "Counterfactual structure.*tuple.*Given.*"):
      _ = counterfactual_model.CounterfactualModel(
          tf.keras.Sequential(),
          [losses.PairwiseMSELoss(),
           losses.PairwiseMSELoss()])

  def testConformWeightsHelperForSingleLoss(self):
    loss = "fake_loss"
    default = 3.0
    weight = 2.0

    # weight value should be used if available otherwise default.
    res = counterfactual_model._conform_weights_to_losses(loss, weight, default)
    self.assertEqual(res, weight)
    res = counterfactual_model._conform_weights_to_losses(loss, None, default)
    self.assertEqual(res, default)

    # Error raised if loss is unnested but weight is not.
    with self.assertRaisesRegex(
        ValueError, "`loss` and `loss_weight`.*do not have matching structures"
        "(.|\n)*type=str str=fake_loss(.|\n)*type=dict str={'k1': 2.0}.*"):
      _ = counterfactual_model._conform_weights_to_losses(
          loss, {"k1": weight}, default)

  def testConformWeightsHelperForDictLoss(self):
    loss = {"k1": "fake_loss1", "k2": "fake_loss2"}
    default = 3.0
    weight = 2.0

    # Values should be broadcast if weight is single element.
    res = counterfactual_model._conform_weights_to_losses(loss, weight, default)
    self.assertDictEqual(res, {"k1": weight, "k2": weight})
    res = counterfactual_model._conform_weights_to_losses(loss, None, default)
    self.assertDictEqual(res, {"k1": default, "k2": default})

    # Dict of weights i matched correctly.
    res = counterfactual_model._conform_weights_to_losses(
        loss, {
            "k1": weight,
            "k2": weight + 1
        }, default)
    self.assertDictEqual(res, {"k1": weight, "k2": weight + 1})

    # Missing values filled in with defaults.
    res = counterfactual_model._conform_weights_to_losses(
        loss, {"k1": weight}, default)
    self.assertDictEqual(res, {"k1": weight, "k2": default})

    # Error raised if weight has unmatched keys.
    with self.assertRaisesRegex(
        ValueError,
        "`loss_weight` contains keys that do not correspond to losses:\n\n"
        "loss: {'k1': 'fake_loss1', 'k2': 'fake_loss2'}\n\n"
        "loss_weight: {'k3': 2.0}"):
      _ = counterfactual_model._conform_weights_to_losses(
          loss, {"k3": weight}, default)

    # Error raised if loss_weight is not a single element or a dict (when weight
    # is a dict).
    with self.assertRaisesRegex(ValueError,
                                "neither a single element nor a dict"):
      _ = counterfactual_model._conform_weights_to_losses(
          loss, [weight, weight], default)

    # Error raised if loss is an invalid dict.
    with self.assertRaisesRegex(
        ValueError, "loss.*loss_weight.*with default "
        "weights.*do not have matching structure"):
      _ = counterfactual_model._conform_weights_to_losses(
          loss, {"k1": {
              "k": weight
          }}, default)

  def testConformWeightsHelperBadLossRaisesError(self):
    bad_loss = ["fake_loss1"]
    # Error raised if loss is not a valid Counterfactual structure.
    with self.assertRaisesRegex(
        TypeError, "loss.*not a recognized "
        "Counterfactual structure.*type.*Given.*list"):
      _ = counterfactual_model._conform_weights_to_losses(bad_loss, None, None)

  def testWeightDefaults(self):
    # Assert single loss is matched by single float of value 1.0
    cf_model = counterfactual_model.CounterfactualModel(
        tf.keras.Sequential(), losses.PairwiseMSELoss())
    self.assertEqual(cf_model._counterfactual_loss_weights, 1.0)

    # Assert multiple losses is matched by multiple floats each of value 1.0.
    cf_model = counterfactual_model.CounterfactualModel(tf.keras.Sequential(), {
        "k1": losses.PairwiseMSELoss(),
        "k2": losses.PairwiseMSELoss()
    })
    self.assertDictEqual(cf_model._counterfactual_loss_weights, {
        "k1": 1.0,
        "k2": 1.0
    })

  def testWeightPartialDefaults(self):
    # Assert single loss is matched by single float of value 1.0
    cf_model = counterfactual_model.CounterfactualModel(
        tf.keras.Sequential(), losses.PairwiseMSELoss())
    self.assertEqual(cf_model._counterfactual_loss_weights, 1.0)

    # Assert missing weights are get a value of 1.0 .
    cf_model = counterfactual_model.CounterfactualModel(
        tf.keras.Sequential(), {
            "k1": losses.PairwiseMSELoss(),
            "k2": losses.PairwiseMSELoss(),
            "k3": losses.PairwiseMSELoss()
        },
        loss_weight={"k2": 2.0})
    self.assertDictEqual(cf_model._counterfactual_loss_weights, {
        "k1": 1.0,
        "k2": 2.0,
        "k3": 1.0
    })

  def testWeightsGetBroadcast(self):
    # Assert all weights are get a value of 1.0 .
    cf_model = counterfactual_model.CounterfactualModel(
        tf.keras.Sequential(), {
            "k1": losses.PairwiseMSELoss(),
            "k2": losses.PairwiseMSELoss()
        },
        loss_weight=2.0)
    self.assertDictEqual(cf_model._counterfactual_loss_weights, {
        "k1": 2.0,
        "k2": 2.0
    })

  def testBadWeightStructureRaisesError(self):
    # Assert bad structure (nested dict) raises an error.
    with self.assertRaisesRegex(
        ValueError, "`loss_weight` contains keys that do not correspond to "
        "losses:.*"):
      _ = counterfactual_model.CounterfactualModel(
          tf.keras.Sequential(), {
              "k1": losses.PairwiseMSELoss(),
              "k2": losses.PairwiseMSELoss()
          },
          loss_weight={"k3": {
              "k": 2.0
          }})

  def testBatchingAndMultiEpoch(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    batch_counter = BatchCounterCallback()
    cf_model.fit(
        self.packed_dataset.batch(2),
        epochs=3,
        verbose=1,
        callbacks=[batch_counter])
    self.assertEqual(
        batch_counter.batch_end_count,
        6)  # 4 examples in dataset, batches of size 2, over 3 epochs

  def testTotalLossForModelSumsoriginalLossAndCompiledLoss(self):
    original_model = get_original_model()
    cf_loss_obj = losses.PairwiseMSELoss()
    compiled_loss_obj = tf.keras.losses.MeanAbsoluteError()
    cf_model = counterfactual_model.CounterfactualModel(original_model,
                                                        cf_loss_obj)
    cf_model.compile(loss=compiled_loss_obj)

    y = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    y_pred_original = tf.constant(
        [[1, 1, 0], [-2, -3, -2]], dtype=tf.float32)
    y_pred_cf = tf.constant([[-1, -1, -1], [-2, -3, -2]], dtype=tf.float32)
    cf_w = tf.constant([5, 7], dtype=tf.float32)
    w = tf.constant([1, 2], dtype=tf.float32)

    total_loss, cf_loss, compiled_loss = cf_model.compute_total_loss(
        y, y_pred, y_pred_original, y_pred_cf, w, cf_w)
    self.assertAllClose(total_loss, cf_loss + compiled_loss)
    self.assertAllClose(cf_loss, cf_loss_obj(y_pred_original, y_pred_cf, cf_w))
    self.assertAllClose(compiled_loss, compiled_loss_obj(y, y_pred, w))

  def testTrainModelMetrics(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(loss="mae", optimizer="adam", metrics=["mae"])
    history = cf_model.fit(self.packed_dataset.batch(1), verbose=1)
    self.assertSetEqual(
        set(["total_loss", "counterfactual_loss", "original_loss", "mae"]),
        set(history.history.keys()))

  def testFittingModelWithoutTrainingDataThrowsError(self):
    original_model = tf.keras.Sequential()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(loss="mae")
    with self.assertRaisesRegex(
        ValueError,
        "Training data must be an instance of CounterfactualPackedInputs.*"):
      _ = cf_model.fit(self.original_x)

  def testCFModelInferenceCallsOriginalModel(self):
    original_model = tf.keras.Sequential()
    original_model_predictions = tf.constant([1.0])
    original_model.call = mock.MagicMock(
        return_value=original_model_predictions)

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compute_total_loss = mock.MagicMock()

    cf_model_predictions = cf_model(self.original_x)
    cf_model.compute_total_loss.assert_not_called()
    original_model.call.assert_called_with(self.original_x, training=None)
    self.assertEmpty(cf_model.losses)
    self.assertAllClose(original_model_predictions, cf_model_predictions)

  def testMetricsForModelEvaluateWithCounterfactualData(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(loss="mae", optimizer="adam")
    output_metrics = cf_model.test_step(
        iter(self.packed_dataset.batch(1)).get_next())
    self.assertSetEqual(
        set(output_metrics.keys()),
        set(["total_loss", "counterfactual_loss", "original_loss"]))
    self.assertAllClose(output_metrics["total_loss"], 174)
    self.assertAllClose(output_metrics["original_loss"], 30)
    self.assertAllClose(output_metrics["counterfactual_loss"],
                        144)  # ((-1-2-3)-(1+2+3))^2

  def testMetricsForModelEvaluateWithNoCounterfactualData(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(loss="mae", optimizer="adam")
    output_metrics = cf_model.test_step(
        iter(self.original_input.batch(1)).get_next())
    self.assertSetEqual(
        set(output_metrics.keys()), set(["total_loss", "original_loss"]))
    self.assertAllClose(output_metrics["total_loss"], 30)
    self.assertAllClose(output_metrics["original_loss"], 30)

  def testHappyPath(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model.compile(loss="mae", optimizer="adam")
    cf_model.fit(self.packed_dataset.batch(2), verbose=1)
    cf_model.evaluate(self.packed_dataset.batch(2), verbose=1)

  def testWithRegularizationInOriginalModel(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
        tf.keras.layers.ActivityRegularization(0.34)
    ])
    cf_model = counterfactual_model.CounterfactualModel(
        copy.deepcopy(original_model), losses.PairwiseMSELoss())

    # Compile with `run_eagerly=True` so that we can evaluate model.losses.
    compile_args = {
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ["acc"],
        "run_eagerly": True
    }
    original_model.compile(**compile_args)
    cf_model.compile(**compile_args)

    _ = original_model.fit(self.original_input.batch(1))
    _ = cf_model.fit(self.packed_dataset.batch(1))

    # We expect only 1 regularization element in the original model's training.
    self.assertLen(original_model.losses, 1)

    # We expect 1 regularization elements in the counterfactual model's
    # training corresponds to the original_model's regularization loss.
    self.assertLen(cf_model.losses, 1)

    # The Counterfactual loss function should be added to
    # `_counterfactual_losses`.
    self.assertIsInstance(cf_model._counterfactual_losses,
                          losses.PairwiseMSELoss)

    # The regularization elements of the original should be the same.
    self.assertListEqual(original_model.losses, cf_model.losses)

  def testComputeCounterfactualLossWithDefaultWeights(self):
    original_model = get_original_model()
    predictions = tf.expand_dims(tf.range(1.0, 51), axis=-1)
    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    # Assert correct inference call and calculated loss.
    y_pred = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    y_pred_cf = tf.constant([[-1, -1, -1], [-2, -2, -2]], dtype=tf.float32)
    cf_w = tf.constant([5, 7], dtype=tf.float32)

    loss = cf_model.compute_counterfactual_loss(
        y_pred, y_pred_cf, cf_w)

    expected_single_counterfactual_loss_weighted = losses.PairwiseMSELoss(
    ).call(y_pred, y_pred_cf, cf_w)
    self.assertAllClose(expected_single_counterfactual_loss_weighted, loss)

  def testComputeCounterfactualLossWithWeights(self):
    original_model = get_original_model()
    predictions = tf.expand_dims(tf.range(1.0, 51), axis=-1)
    loss_weight = 2.3  # Arbitrary value.

    # Mock original_model's call function.
    original_model.call = mock.MagicMock(return_value=predictions)

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss(), loss_weight=loss_weight)

    # Assert correct inference call and calculated loss.
    y_pred = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    y_pred_cf = tf.constant([[-1, -1, -1], [-2, -2, -2]], dtype=tf.float32)
    cf_w = tf.constant([5, 7], dtype=tf.float32)

    loss = cf_model.compute_counterfactual_loss(
        y_pred, y_pred_cf, cf_w)

    expected_single_counterfactual_loss_weighted = losses.PairwiseMSELoss(
    ).call(y_pred, y_pred_cf, cf_w)
    self.assertAllClose(
        expected_single_counterfactual_loss_weighted * loss_weight, loss)

  def testComputeCounterfactualLossPassesInputsThrough(self):
    y_pred = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    y_pred_cf = tf.constant([[-1, -1, -1], [-2, -2, -2]], dtype=tf.float32)
    cf_w = tf.constant([5, 7], dtype=tf.float32)

    original_model = get_original_model()

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())
    cf_model._counterfactual_losses = mock.MagicMock()

    _ = cf_model.compute_counterfactual_loss(y_pred, y_pred_cf, cf_w)
    cf_model._counterfactual_losses.assert_called_once_with(
        counterfactual=y_pred_cf, original=y_pred, sample_weight=cf_w)
    cf_model._counterfactual_losses.reset_mock()

  def testComputeCounterfactualLoss(self):
    original_model = tf.keras.Sequential()

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    y_pred = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    y_pred_cf = tf.constant([[-1, -1, -1], [-2, -2, -2]], dtype=tf.float32)
    cf_w = tf.constant([5, 7], dtype=tf.float32)

    # Assert correct inference call and calculated loss.
    loss = cf_model.compute_counterfactual_loss(
        y_pred, y_pred_cf, cf_w)
    expected_single_counterfactual_loss = losses.PairwiseMSELoss().call(
        y_pred, y_pred_cf, cf_w)
    self.assertAllClose(expected_single_counterfactual_loss, loss)

  def testCustomModelWithFunctionalRaisesErrorIfNoSkipInit(self):

    class CustomModel(tf.keras.Model):

      def __init__(self, *args, **kwargs):
        # Explicitly don't pass in other kwargs. This causes problems when
        # model is used via the Functional API.
        super(CustomModel, self).__init__(kwargs["inputs"], kwargs["outputs"])

    inputs = tf.keras.Input(1)
    outputs = tf.keras.layers.Dense(1, activation="softmax")(inputs)
    original_model = CustomModel(inputs=inputs, outputs=outputs)

    class CustomCounterfactualModel(counterfactual_model.CounterfactualModel,
                                    CustomModel):
      pass

    with self.assertRaisesRegex(
        ValueError,
        "problem initializing the CounterfactualModel subclass instance"):
      _ = CustomCounterfactualModel(original_model, losses.PairwiseMSELoss())

  def testSaveOriginalModelThroughCFModelThenRewrap(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    cf_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = cf_model.fit(self.packed_dataset.batch(1))

    # Create temp directory and save original model.
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      cf_model.save_original_model(path)

      loaded_model = tf.keras.models.load_model(path)

    cf_model = counterfactual_model.CounterfactualModel(
        loaded_model, losses.PairwiseMSELoss())
    cf_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    _ = cf_model.fit(self.packed_dataset.batch(2))

    # Evaluate with counterfactual_data.
    cf_model.evaluate(self.packed_dataset.batch(2))

    # Evaluate and run inference without counterfactual_data.
    cf_model.evaluate(self.original_input.batch(2))
    cf_model.predict(self.original_input.batch(2))

  def testSaveCFModelAndLoadWeights(self):
    original_model = get_original_model()
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    cf_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    _ = cf_model.fit(self.packed_dataset.batch(1))

    # Create temp directory and save orginal weights.
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "saved_model")
      cf_model.save_weights(path)

      original_model = tf.keras.Sequential([
          tf.keras.layers.Dense(1, activation="softmax"),
      ])

      # Load model with saved weights.
      loaded_model = counterfactual_model.CounterfactualModel(
          original_model, losses.PairwiseMSELoss())
      loaded_model.compile(
          loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
      loaded_model.load_weights(path)

    _ = cf_model.fit(self.packed_dataset.batch(2))

    # Evaluate with counterfactual_data.
    cf_model.evaluate(self.packed_dataset.batch(2))

    # Evaluate and run inference without counterfactual_data.
    cf_model.evaluate(self.original_input.batch(2))
    cf_model.predict(self.original_input.batch(2))

  def testSerialization(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])
    loss_weight = 2.3  # Arbitrary value.
    model_name = "custom_model_name"  # Arbitrary name.
    cf_model = counterfactual_model.CounterfactualModel(
        original_model,
        losses.PairwiseMSELoss(),
        loss_weight=loss_weight,
        name=model_name)

    serialized_model = tf.keras.utils.serialize_keras_object(cf_model)
    deserialized_model = tf.keras.layers.deserialize(serialized_model)

    self.assertIsInstance(deserialized_model,
                          counterfactual_model.CounterfactualModel)
    self.assertIsInstance(deserialized_model._counterfactual_losses,
                          losses.PairwiseMSELoss)
    self.assertEqual(deserialized_model._counterfactual_loss_weights,
                     loss_weight)
    self.assertEqual(deserialized_model.name, model_name)

  def testSerializationForMultipleApplications(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    # Arbitrary losses.
    loss = {"k1": losses.PairwiseMSELoss(), "k2": losses.PairwiseMSELoss()}
    # Arbitrary values.
    loss_weight = {"k1": 3.4, "k2": 2.9}
    model_name = "custom_model_name"  # Arbitrary name.
    cf_model = counterfactual_model.CounterfactualModel(
        original_model, loss, loss_weight, name=model_name)

    serialized_model = tf.keras.utils.serialize_keras_object(cf_model)
    deserialized_model = tf.keras.layers.deserialize(serialized_model)

    self.assertIsInstance(deserialized_model,
                          counterfactual_model.CounterfactualModel)

    self.assertDictEqual(deserialized_model._counterfactual_losses, loss)
    self.assertDictEqual(deserialized_model._counterfactual_loss_weights,
                         loss_weight)
    self.assertEqual(deserialized_model.name, model_name)

  def testConfig(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    config = cf_model.get_config()

    self.assertSetEqual(
        set(config.keys()),
        set(["original_model", "loss", "loss_weight", "name"]))

    # Test building the model from the config.
    model_from_config = counterfactual_model.CounterfactualModel.from_config(
        config)
    self.assertIsInstance(model_from_config,
                          counterfactual_model.CounterfactualModel)
    self.assertIsInstance(model_from_config.original_model, tf.keras.Sequential)

  def testGetConfigForMultipleApplications(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    # Arbitrary losses.
    loss = {"k1": losses.PairwiseMSELoss(), "k2": losses.PairwiseMSELoss()}
    # Arbitrary values.
    loss_weight = {"k1": 3.4, "k2": 2.9}
    cf_model = counterfactual_model.CounterfactualModel(original_model, loss,
                                                        loss_weight)

    config = cf_model.get_config()

    self.assertSetEqual(
        set(config.keys()),
        set(["original_model", "loss", "loss_weight", "name"]))

    self.assertDictEqual(config["loss"], loss)
    self.assertDictEqual(config["loss_weight"], loss_weight)

  def testBuildingConfigForMultipleApplications(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    # Arbitrary losses.
    loss = {"k1": losses.PairwiseMSELoss(), "k2": losses.PairwiseMSELoss()}
    # Arbitrary values.
    loss_weight = {"k1": 3.4, "k2": 2.9}
    cf_model = counterfactual_model.CounterfactualModel(original_model, loss,
                                                        loss_weight)

    model_from_config = counterfactual_model.CounterfactualModel.from_config(
        cf_model.get_config())
    self.assertIsInstance(model_from_config,
                          counterfactual_model.CounterfactualModel)
    self.assertIsInstance(model_from_config.original_model, tf.keras.Sequential)

    self.assertDictEqual(model_from_config._counterfactual_losses, loss)
    self.assertDictEqual(model_from_config._counterfactual_loss_weights,
                         loss_weight)

  def testGetConfigWithCustomBaseImplementation(self):

    class CustomModel(tf.keras.Model):

      def __init__(self, val, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.val = val

      def get_config(self):
        return {"val": self.val}

    class CustomCounterfactualModel(counterfactual_model.CounterfactualModel,
                                    CustomModel):
      pass  # No additional implementation needed.

    original_val = 4  # Arbitrary value passed in.
    original_model = CustomModel(original_val)

    counterfactual_model_val = 5  # Different arbitrary value passed in.
    cf_model = CustomCounterfactualModel(
        original_model=original_model,
        loss=losses.PairwiseMSELoss(),
        val=counterfactual_model_val)
    self.assertEqual(cf_model.val, counterfactual_model_val)

    config = cf_model.get_config()

    self.assertSetEqual(
        set(config.keys()),
        set(["original_model", "loss", "loss_weight", "name", "val"]))
    self.assertEqual(config["val"], cf_model.val)
    self.assertEqual(config["original_model"].val, original_model.val)

  def testBuildingConfigWithCustomBaseImplementation(self):

    class CustomModel(tf.keras.Model):

      def __init__(self, val, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.val = val

      def get_config(self):
        return {"val": self.val}

    class CustomCounterfactualModel(counterfactual_model.CounterfactualModel,
                                    CustomModel):
      pass  # No additional implementation needed.

    original_val = 4  # Arbitrary value passed in.
    original_model = CustomModel(original_val)

    counterfactual_model_val = 5  # Different arbitrary value passed in.
    cf_model = CustomCounterfactualModel(
        original_model=original_model,
        loss=losses.PairwiseMSELoss(),
        val=counterfactual_model_val)
    self.assertEqual(cf_model.val, counterfactual_model_val)

    config = cf_model.get_config()
    model_from_config = CustomCounterfactualModel.from_config(config)
    self.assertIsInstance(model_from_config, CustomCounterfactualModel)
    self.assertIsInstance(model_from_config.original_model, CustomModel)
    self.assertEqual(model_from_config.val, counterfactual_model_val)
    self.assertEqual(model_from_config.original_model.val, original_val)

  def testGetConfigFromOriginalModel(self):
    original_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="softmax"),
    ])

    cf_model = counterfactual_model.CounterfactualModel(
        original_model, losses.PairwiseMSELoss())

    self.assertEqual(
        set(cf_model.get_config().keys()),
        set(["loss", "loss_weight", "name", "original_model"]))

if __name__ == "__main__":
  tf.test.main()
