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

"""Util methods for the MinDiff Keras colab."""

import os
import os.path
import tempfile

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'


def download_and_process_civil_comments_data():
  """Download and process the civil comments dataset into a Pandas DataFrame."""

  # Download data.
  toxicity_data_url = 'https://storage.googleapis.com/civil_comments_dataset/'
  train_csv_file = tf.keras.utils.get_file(
      'train_df_processed.csv', toxicity_data_url + 'train_df_processed.csv')
  validate_csv_file = tf.keras.utils.get_file(
      'validate_df_processed.csv',
      toxicity_data_url + 'validate_df_processed.csv')

  # Get validation data as TFRecords.
  validate_tfrecord_file = tf.keras.utils.get_file(
      'validate_tf_processed.tfrecord',
      toxicity_data_url + 'validate_tf_processed.tfrecord')

  # Read data into Pandas DataFrame.
  data_train = pd.read_csv(train_csv_file)
  data_validate = pd.read_csv(validate_csv_file)

  # Fix type interpretation.
  data_train[TEXT_FEATURE] = data_train[TEXT_FEATURE].astype(str)
  data_validate[TEXT_FEATURE] = data_validate[TEXT_FEATURE].astype(str)

  # Shape labels to match output.
  labels_train = data_train[LABEL].values.reshape(-1, 1) * 1.0
  labels_validate = data_validate[LABEL].values.reshape(-1, 1) * 1.0

  return data_train, data_validate, validate_tfrecord_file, labels_train, labels_validate


def _create_embedding_layer(hub_url):
  return hub.KerasLayer(
      hub_url, output_shape=[128], input_shape=[], dtype=tf.string)


# pylint does not understand the default arg values, thinks they are empty []
# pylint:disable=dangerous-default-value
def create_keras_sequential_model(
    hub_url='https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1',
    cnn_filter_sizes=[128, 128, 128],
    cnn_kernel_sizes=[5, 5, 5],
    cnn_pooling_sizes=[5, 5, 40]):
  """Create baseline keras sequential model."""

  model = tf.keras.Sequential()

  # Embedding layer.
  hub_layer = _create_embedding_layer(hub_url)
  model.add(hub_layer)
  model.add(tf.keras.layers.Reshape((1, 128)))

  # Convolution layers.
  for filter_size, kernel_size, pool_size in zip(cnn_filter_sizes,
                                                 cnn_kernel_sizes,
                                                 cnn_pooling_sizes):
    model.add(
        tf.keras.layers.Conv1D(
            filter_size, kernel_size, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size, padding='same'))

  # Flatten, fully connected, and output layers.
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  return model


def get_eval_results(model_location,
                     base_dir,
                     eval_subdir,
                     validate_tfrecord_file,
                     slice_selection='religion',
                     compute_confidence_intervals=True):
  """Get Fairness Indicators eval results."""
  tfma_eval_result_path = os.path.join(base_dir, 'tfma_eval_result')

  # Define slices that you want the evaluation to run on.
  eval_config = text_format.Parse(
      """
    model_specs {
     label_key: '%s'
   }
   metrics_specs {
     metrics {class_name: "AUC"}
     metrics {class_name: "ExampleCount"}
     metrics {class_name: "Accuracy"}
     metrics {
        class_name: "FairnessIndicators"
        config: '{"thresholds": [0.4, 0.4125, 0.425, 0.4375, 0.45, 0.4675, 0.475, 0.4875, 0.5]}'
     }
   }
   slicing_specs {
     feature_keys: '%s'
   }
   slicing_specs {}
   options {
       compute_confidence_intervals { value: %s }
       disabled_outputs{values: "analysis"}
   }
   """ % (LABEL,
          slice_selection, 'true' if compute_confidence_intervals else 'false'),
      tfma.EvalConfig())

  base_dir = tempfile.mkdtemp(prefix='saved_eval_results')
  tfma_eval_result_path = os.path.join(base_dir, eval_subdir)

  eval_shared_model = tfma.default_eval_shared_model(
      eval_saved_model_path=model_location, tags=[tf.saved_model.SERVING])

  # Run the fairness evaluation.
  return tfma.run_model_analysis(
      eval_shared_model=eval_shared_model,
      data_location=validate_tfrecord_file,
      file_format='tfrecords',
      eval_config=eval_config,
      output_path=tfma_eval_result_path,
      extractors=None)
