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

"""Tests to ensure there are no unexpected API regressions.

These tests should be a comprehensive list of the public elements in the API.
Anything not listed here should be considered private and subject to change at
any time.
These tests are basic existence assertions. They ensure that the symbols are
available but do nothing to verify that content (signature, attributes, methods,
etc..) remains the same.
"""

import tensorflow as tf

from tensorflow_model_remediation import min_diff


class KerasAPITest(tf.test.TestCase):

  def testMinDiffModelAPI(self):
    _ = min_diff.keras.MinDiffModel
    _ = min_diff.keras.models.MinDiffModel

  def testInputUtilsAPI(self):
    _ = min_diff.keras.utils.build_min_diff_dataset
    _ = min_diff.keras.utils.MinDiffPackedInputs
    _ = min_diff.keras.utils.pack_min_diff_data
    _ = min_diff.keras.utils.unpack_min_diff_data
    _ = min_diff.keras.utils.unpack_original_inputs

  def testStructureUtilsAPI(self):
    _ = min_diff.keras.utils.validate_min_diff_structure


class LossesAPITest(tf.test.TestCase):

  def testLossesAPI(self):
    _ = min_diff.losses.MinDiffLoss
    _ = min_diff.losses.AbsoluteCorrelationLoss
    _ = min_diff.losses.MMDLoss
    _ = min_diff.losses.AdjustedMMDLoss

  def testKernelsAPI(self):
    _ = min_diff.losses.MinDiffKernel
    _ = min_diff.losses.kernels.MinDiffKernel
    _ = min_diff.losses.GaussianKernel
    _ = min_diff.losses.kernels.GaussianKernel
    _ = min_diff.losses.LaplacianKernel
    _ = min_diff.losses.kernels.LaplacianKernel


if __name__ == "__main__":
  tf.test.main()
