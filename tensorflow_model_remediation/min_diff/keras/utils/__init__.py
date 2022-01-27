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

"""Utility functions and classes for integrating MinDiff."""

from tensorflow_model_remediation.min_diff.keras.utils.input_utils import build_min_diff_dataset
from tensorflow_model_remediation.min_diff.keras.utils.input_utils import MinDiffPackedInputs
from tensorflow_model_remediation.min_diff.keras.utils.input_utils import pack_min_diff_data
from tensorflow_model_remediation.min_diff.keras.utils.input_utils import unpack_min_diff_data
from tensorflow_model_remediation.min_diff.keras.utils.input_utils import unpack_original_inputs

from tensorflow_model_remediation.min_diff.keras.utils.structure_utils import validate_min_diff_structure
