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

"""Utility functions and classes for integrating Counterfactual."""

from tensorflow_model_remediation.counterfactual.keras.utils.input_utils import build_counterfactual_data
from tensorflow_model_remediation.counterfactual.keras.utils.input_utils import CounterfactualPackedInputs
from tensorflow_model_remediation.counterfactual.keras.utils.input_utils import pack_counterfactual_data

from tensorflow_model_remediation.counterfactual.keras.utils.structure_utils import validate_counterfactual_structure
