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

from tensorflow_model_remediation import counterfactual


class KerasAPITest(tf.test.TestCase):

  def testCounterfactualModelAPI(self):
    _ = counterfactual.keras.CounterfactualModel
    _ = counterfactual.keras.models.CounterfactualModel

  def testInputUtilsAPI(self):
    _ = counterfactual.keras.utils.build_counterfactual_data
    _ = counterfactual.keras.utils.CounterfactualPackedInputs
    _ = counterfactual.keras.utils.pack_counterfactual_data

  def testStructureUtilsAPI(self):
    _ = counterfactual.keras.utils.validate_counterfactual_structure


class LossesAPITest(tf.test.TestCase):

  def testLossesAPI(self):
    _ = counterfactual.losses.CounterfactualLoss
    _ = counterfactual.losses.PairwiseAbsoluteDifferenceLoss
    _ = counterfactual.losses.PairwiseMSELoss
    _ = counterfactual.losses.PairwiseCosineLoss


if __name__ == "__main__":
  tf.test.main()
