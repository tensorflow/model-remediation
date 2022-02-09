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

"""Useful constants for typing the TensorFlow Model Remediation library."""

from typing import Callable, Dict, Tuple, Union

import tensorflow as tf

TensorType = Union[tf.Tensor, tf.SparseTensor]
FeatureOrLabelType = Union[TensorType, Dict[str, TensorType]]
FeatureOrLabelTransformType = Callable[[FeatureOrLabelType], FeatureOrLabelType]
InputFnType = Callable[[], Tuple[FeatureOrLabelType, FeatureOrLabelType]]
OptimizerType = tf.keras.optimizers.Optimizer
# `complex` is used for typing to account for values that would normally be
# included within numbers.Number (e.g. int and float). For additional
# information on why this is used please refer to the below link:
# https://www.python.org/dev/peps/pep-0484/#the-numeric-tower
TensorOrScalar = Union[complex, TensorType]
TensorTransformType = Callable[[TensorType], TensorType]
TrainOpFnType = Callable[[TensorType], OptimizerType]
