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

"""Tests for tensorflow_model_remediation.tools.tutorials_utils.uci.utils."""

import unittest.mock as mock
import pandas as pd

import tensorflow as tf

from tensorflow_model_remediation.tools.tutorials_utils.uci import utils


class UCIDataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    utils._uci_full_dataframes = {}  # Clear any caches.

  def tearDown(self):
    super().tearDown()
    utils._uci_full_dataframes = {}  # Clear any caches.

  @mock.patch('pandas.read_csv', autospec=True)
  def testGetTrainUciDataAsDefault(self, mock_read_csv):
    _ = utils.get_uci_min_diff_datasets()
    mock_read_csv.assert_called_once_with(
        utils._UCI_DATA_URL_TEMPLATE.format('data'),
        names=utils._UCI_COLUMN_NAMES,
        header=None)

  @mock.patch('pandas.read_csv', autospec=True)
  def testGetTrainUciData(self, mock_read_csv):
    _ = utils.get_uci_min_diff_datasets(split='train')
    mock_read_csv.assert_called_once_with(
        utils._UCI_DATA_URL_TEMPLATE.format('data'),
        names=utils._UCI_COLUMN_NAMES,
        header=None)

  @mock.patch('pandas.read_csv', autospec=True)
  def testGetTestUciData(self, mock_read_csv):
    utils.get_uci_min_diff_datasets(split='test')
    mock_read_csv.assert_called_once_with(
        utils._UCI_DATA_URL_TEMPLATE.format('test'),
        names=utils._UCI_COLUMN_NAMES,
        header=None)

  @mock.patch('pandas.read_csv', autospec=True)
  def testGetSampledUciData(self, mock_read_csv):
    mock_df = mock.MagicMock()
    mock_df.sample = mock.MagicMock()
    mock_read_csv.return_value = mock_df
    sample = 0.45  # Arbitrary number.
    utils.get_uci_min_diff_datasets(sample=sample)
    mock_read_csv.assert_called_once()
    mock_df.sample.assert_called_once_with(
        frac=sample, replace=True, random_state=1)

  @mock.patch('pandas.read_csv', autospec=True)
  def testUciDataCached(self, mock_read_csv):
    utils.get_uci_min_diff_datasets()
    mock_read_csv.assert_called_once()

    # pandas.read_csv should not be called again for the same split ('train')
    mock_read_csv.reset_mock()
    utils.get_uci_min_diff_datasets()
    mock_read_csv.assert_not_called()

    # pandas.read_csv should be called again for a different split
    mock_read_csv.reset_mock()
    utils.get_uci_min_diff_datasets(split='test')
    mock_read_csv.assert_called_once()

    # pandas.read_csv should not be called again (both splits have been cached)
    mock_read_csv.reset_mock()
    utils.get_uci_min_diff_datasets(split='train')
    utils.get_uci_min_diff_datasets(split='test')
    mock_read_csv.assert_not_called()

  def testGetUciDataWithBadSplitRaisesError(self):
    with self.assertRaisesRegex(ValueError,
                                'split must be.*train.*test.*given.*bad_split'):
      utils.get_uci_min_diff_datasets('bad_split')

  def testConvertToDataset(self):
    expected_vals = range(10)  # Arbitrary values
    expected_labels = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0]  # Arbitrary labels.
    data = {
        'col': pd.Series(expected_vals),
        'target': pd.Series(expected_labels)
    }
    df = pd.DataFrame(data)
    dataset = utils.df_to_dataset(df)
    vals, labels = zip(*[(val, label.numpy()) for val, label in dataset])

    # Assert values are all dicts with exactly one column.
    for val in vals:
      self.assertSetEqual(set(val.keys()), set(['col']))
    vals = [val['col'].numpy() for val in vals]

    self.assertAllClose(vals, expected_vals)
    self.assertAllClose(labels, expected_labels)

  def testConvertToDatasetWithShuffle(self):
    expected_vals = range(10)  # Arbitrary values
    expected_labels = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0]  # Arbitrary labels.
    data = {
        'col': pd.Series(expected_vals),
        'target': pd.Series(expected_labels)
    }
    df = pd.DataFrame(data)
    dataset = utils.df_to_dataset(df, shuffle=True)
    vals, labels = zip(*[(val, label.numpy()) for val, label in dataset])

    # Assert values are all dicts with exactly one column.
    for val in vals:
      self.assertSetEqual(set(val.keys()), set(['col']))
    vals = [val['col'].numpy() for val in vals]

    # These values should *NOT* be close because vals should be out of order
    # since we set shuffle=True. Note that it seems like the tests provide a
    # consistent seed so we don't have to worry about getting unlucky. If this
    # changes then we can look into explicitly setting the a seed.
    self.assertNotAllClose(vals, expected_vals)
    # Assert that the contents are the same, just reordered
    self.assertAllClose(sorted(vals), sorted(expected_vals))
    self.assertAllClose(sorted(labels), sorted(expected_labels))


class MinDiffDatasetsTest(tf.test.TestCase):

  @mock.patch('tensorflow_model_remediation.tools.tutorials_utils'
              '.uci.utils.df_to_dataset',
              autospec=True)
  @mock.patch('tensorflow_model_remediation.tools.tutorials_utils'
              '.uci.utils.get_uci_data',
              autospec=True)
  def testGetMinDiffDatasets(self, mock_get_uci_data, mock_df_to_dataset):
    mock_md_ds1 = mock.MagicMock()
    mock_md_ds1.batch.return_value = 'batched_d1'
    mock_md_ds2 = mock.MagicMock()
    mock_md_ds2.batch.return_value = 'batched_d2'
    mock_df_to_dataset.side_effect = ['og_ds', mock_md_ds1, mock_md_ds2]

    sample = 0.56  # Arbitrary sample value.
    split = 'fake_split'
    original_batch_size = 19  # Arbitrary size.
    min_diff_batch_size = 23  # Different arbitrary size.
    res = utils.get_uci_min_diff_datasets(
        split=split,
        sample=sample,
        original_batch_size=original_batch_size,
        min_diff_batch_size=min_diff_batch_size)

    # Assert outputs come from the right place.
    expected_res = ('og_ds', 'batched_d1', 'batched_d2')
    self.assertEqual(res, expected_res)

    # Assert proper calls have been made.
    self.assertEqual(mock_get_uci_data.call_count, 2)
    mock_get_uci_data.assert_has_calls(
        [mock.call(split=split, sample=sample),
         mock.call(split=split)],
        any_order=True)

    self.assertEqual(mock_df_to_dataset.call_count, 3)
    mock_df_to_dataset.assert_has_calls([
        mock.call(mock.ANY, shuffle=True, batch_size=original_batch_size),
        mock.call(mock.ANY, shuffle=True),
        mock.call(mock.ANY, shuffle=True),
    ])
    mock_md_ds1.batch.assert_called_once_with(
        min_diff_batch_size, drop_remainder=True)
    mock_md_ds2.batch.assert_called_once_with(
        min_diff_batch_size, drop_remainder=True)

  @mock.patch('tensorflow_model_remediation.tools.tutorials_utils'
              '.uci.utils.get_uci_data',
              autospec=True)
  def testGetMinDiffDatasetsDefaults(self, mock_get_uci_data):
    _ = utils.get_uci_min_diff_datasets()

    mock_get_uci_data.assert_has_calls(
        [mock.call(split='train', sample=None),
         mock.call(split='train')],
        any_order=True)

  @mock.patch('tensorflow_model_remediation.min_diff.keras.utils.'
              'pack_min_diff_data',
              autospec=True)
  @mock.patch('tensorflow_model_remediation.tools.tutorials_utils'
              '.uci.utils.get_uci_min_diff_datasets',
              autospec=True)
  def testGetUciWithMinDiffDataset(self, mock_get_uci_min_diff_datasets,
                                   mock_pack_data):
    mock_get_uci_min_diff_datasets.return_value = ('og', 'md1', 'md2')
    mock_pack_data.return_value = 'packed_data'

    sample = 0.56  # Arbitrary sample value.
    split = 'fake_split'
    res = utils.get_uci_with_min_diff_dataset(split=split, sample=sample)

    mock_get_uci_min_diff_datasets.assert_called_once_with(
        split=split, sample=sample)
    mock_pack_data.assert_called_once_with(
        original_dataset='og',
        nonsensitive_group_dataset='md1',
        sensitive_group_dataset='md2')
    self.assertEqual(res, 'packed_data')

  @mock.patch('tensorflow_model_remediation.min_diff.keras.utils.'
              'pack_min_diff_data',
              autospec=True)
  @mock.patch('tensorflow_model_remediation.tools.tutorials_utils'
              '.uci.utils.get_uci_min_diff_datasets',
              autospec=True)
  def testGetUciWithMinDiffDatasetDefaults(self, mock_get_uci_min_diff_datasets,
                                           mock_pack_data):
    mock_ds = mock.MagicMock()
    mock_get_uci_min_diff_datasets.return_value = (mock_ds, mock_ds, mock_ds)
    _ = utils.get_uci_with_min_diff_dataset()

    mock_get_uci_min_diff_datasets.assert_called_once_with(
        split='train', sample=None)


class UCIModelTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.read_csv_patch = mock.patch('pandas.read_csv', autospec=True)
    mock_read_csv = self.read_csv_patch.start()
    mock_data = {  # All values are realistic but arbitrary.
        'age': pd.Series([25]),
        'workclass': pd.Series(['Private']),
        'fnlwgt': pd.Series([12456]),
        'education': pd.Series(['Bachelors']),
        'education-num': pd.Series([13]),
        'marital-status': pd.Series(['Never-married']),
        'race': pd.Series(['White']),
        'occupation': pd.Series(['Tech-support']),
        'relationship': pd.Series(['Husband']),
        'sex': pd.Series(['Male']),
        'capital-gain': pd.Series([1304]),
        'capital-loss': pd.Series([0]),
        'hours-per-week': pd.Series([40]),
        'native-country': pd.Series(['United-States']),
        'income': pd.Series(['>50K']),
    }
    mock_read_csv.return_value = pd.DataFrame(mock_data)

  def tearDown(self):
    super().tearDown()
    self.read_csv_patch.stop()

  def testModelStructure(self):
    model = utils.get_uci_model()
    self.assertIsInstance(model, tf.keras.Model)
    expected_inputs = utils._UCI_COLUMN_NAMES.copy()
    expected_inputs.remove('income')
    expected_inputs.remove('race')
    expected_inputs.remove('fnlwgt')
    self.assertSetEqual(
        set([layer.name for layer in model.inputs]), set(expected_inputs))
    self.assertIsInstance(model.layers[-1], tf.keras.layers.Dense)

  def testModelStructureWithCustomClass(self):

    class CustomClass(tf.keras.Model):
      pass  # No additional implementation needed for this test.

    model = utils.get_uci_model(model_class=CustomClass)
    self.assertIsInstance(model, CustomClass)
    expected_inputs = utils._UCI_COLUMN_NAMES.copy()
    expected_inputs.remove('income')
    expected_inputs.remove('race')
    expected_inputs.remove('fnlwgt')
    self.assertSetEqual(
        set([layer.name for layer in model.inputs]), set(expected_inputs))
    self.assertIsInstance(model.layers[-1], tf.keras.layers.Dense)

  def testModelRunsOnUCIData(self):
    model = utils.get_uci_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    data = utils.get_uci_data()
    print('DATA:', data)
    dataset = utils.df_to_dataset(data, batch_size=64)

    # Model can train on UCI data
    model.fit(dataset, epochs=1)

    # Model can evaluate on UCI data
    model.evaluate(dataset)

  def testModelRaisesErrorForBadClass(self):
    with self.assertRaisesRegex(TypeError,
                                'must be a class.*given.*not_a_class'):
      _ = utils.get_uci_model(model_class='not_a_class')

    with self.assertRaisesRegex(
        TypeError, 'must be a subclass of.*keras.Model.*given.*str'):
      _ = utils.get_uci_model(model_class=type('bad_class'))

    with self.assertRaisesRegex(
        TypeError, 'must support the Functional API.*cannot be a subclass '
        'of.*Sequential.*given.*Sequential'):
      _ = utils.get_uci_model(model_class=tf.keras.Sequential)


if __name__ == '__main__':
  tf.test.main()
