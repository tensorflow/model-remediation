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

"""Tests for min_diff_keras_util functions."""

import csv
import os
import tempfile
import unittest.mock as mock

import tensorflow as tf
from tensorflow_model_remediation.tools.tutorials_utils import min_diff_keras_utils


class UtilTest(tf.test.TestCase):

  def _create_example_csv(self, use_fake_embedding=False):
    header = [
        'comment_text',
        'toxicity',
        'heterosexual',
        'homosexual_gay_or_lesbian',
        'bisexual',
        'other_sexual_orientation',
        'male',
        'female',
        'transgender',
        'other_gender',
        'christian',
        'jewish',
        'muslim',
        'hindu',
        'buddhist',
        'atheist',
        'other_religion',
        'black',
        'white',
        'asian',
        'latino',
        'other_race_or_ethnicity',
        'physical_disability',
        'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness',
        'other_disability',
    ]
    example = [
        'comment 1' if not use_fake_embedding else 0.35,
        0.1,
        # sexual orientation
        0.1,
        0.1,
        0.5,
        0.1,
        # gender
        0.1,
        0.2,
        0.3,
        0.4,
        # religion
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        # race or ethnicity
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        # disability
        0.6,
        0.7,
        0.8,
        1.0,
    ]
    empty_comment_example = [
        '' if not use_fake_embedding else 0.35,
        0.1,
        0.1,
        0.1,
        0.5,
        0.1,
        0.1,
        0.2,
        0.3,
        0.4,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
    ]
    return [header, example, empty_comment_example]

  def _write_csv(self, examples):
    filename = os.path.join(tempfile.mkdtemp(), 'input.csv')
    with open(filename, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile, delimiter=',')
      for example in examples:
        csvwriter.writerow(example)

    return filename

  @mock.patch(
      'tensorflow_model_remediation.tools.tutorials_utils.min_diff_keras_utils._create_embedding_layer',
      autospec=True)
  @mock.patch('tensorflow.keras.utils.get_file', autospec=True)
  def test_download_and_process_civil_comments_data_and_create_model(
      self, mock_get_file, mock__create_embedding_layer):

    # First test download_and_process_civil_comments_data.  Mock out the
    # download.
    filename = self._write_csv(
        self._create_example_csv(use_fake_embedding=True))
    mock_get_file.return_value = filename
    data_train, _, _, labels_train, _ = min_diff_keras_utils.download_and_process_civil_comments_data(
    )

    self.assertEqual(mock_get_file.call_count, 3)

    # Undo the string interpretation of the text_feature, since we are mocking
    # out the embedding layer in the following model testing.
    data_train[min_diff_keras_utils.TEXT_FEATURE] = data_train[
        min_diff_keras_utils.TEXT_FEATURE].astype(float)

    # Now use that data to test create_keras_sequential_model.
    mock__create_embedding_layer.return_value = tf.keras.layers.Dense(units=128)

    model = min_diff_keras_utils.create_keras_sequential_model(hub_url='')

    # Sanity check that you have a valid model by training it and predicting.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(
        x=data_train['comment_text'], y=labels_train, batch_size=1, epochs=1)
    result = model.predict([[0.1]])
    self.assertTrue(result[0][0] < 1 and result[0][0] > 0)


if __name__ == '__main__':
  tf.test.main()
