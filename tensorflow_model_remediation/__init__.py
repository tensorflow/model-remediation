# coding=utf-8
# Copyright 2020 Google LLC.
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

# pyformat: disable
"""TensorFlow Model Remediation Library for addressing concerns in machine learning models.

The library will ultimately contain a collection of techniques for addressing
a wide range of concerns. For now, it has a single technique, MinDiff, which
can help reduce performance gaps between example subgroups.

Other Documentation:

* [Overview](https://www.tensorflow.org/responsible_ai/model_remediation)
* [MinDiff Requirements](https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/guide/requirements)
* [Tutorial](https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras)
"""
# pyformat: enable

from tensorflow_model_remediation import common
from tensorflow_model_remediation import min_diff

from tensorflow_model_remediation.version import __version__
