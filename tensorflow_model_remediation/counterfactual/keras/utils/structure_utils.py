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

"""Utils to validate and manipulate Counterfactual structures (single elems or dicts).

This module is a customized version of `tf.nest` with the main difference being
that tuples are considered single elements rather than structs that can be
unpacked.
"""

import tensorflow as tf


def _flatten_counterfactual_structure(struct, run_validation=False):
  # pyformat: disable
  """Flattens a Counterfactual structure after optionally validating it.

  Arguments:
    struct: structure to be flattened. Must be a single element (including a
      tuple) or an unnested dict.
    run_validation: Boolean indicating whether to run validation. If `True`
      `validate_counterfactual_structure` will be called on `struct`.

  Has similar behavior to `tf.nest.flatten` with the exception that tuples will
  be considered as single elements instead of structures to be flattened. See
  `tf.nest.flatten` documentation for additional details on behavior.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    ValueError: If struct is not a valid Counterfactual structure (a single
      element including a tuple, or a dict).
  """
  # pyformat: enable

  if run_validation:
    validate_counterfactual_structure(struct)

  if isinstance(struct, dict):
    return [struct[key] for key in sorted(struct.keys())]

  return [struct]  # Wrap in a list if not nested.


def _is_counterfactual_element(element, element_type=None):
  """Returns `True` if `element` is a Counterfactual and `False` otherwise."""
  if element_type is not None:
    # Check that element_type is a valid Python type.
    if not isinstance(element_type, type):
      raise TypeError(
          "`element_type` should be a class corresponding to expected type. "
          f"Instead an object instance was given: {element_type}")
    # Return False if element is of the wrong type (if indicated).
    if not isinstance(element, element_type):
      return False

  is_single_element = not tf.nest.is_nested(element)
  is_tuple = isinstance(element, tuple)
  return is_single_element or is_tuple


def validate_counterfactual_structure(struct,
                                      struct_name="struct",
                                      element_type=None):
  # pyformat: disable
  """Validates that `struct` is a valid Counterfactual structure.

  Arguments:
    struct: Structure that will be validated.
    struct_name: Name of struct, used only for error messages.
    element_type: Type of elements. If `None`, types are not checked.

  A `struct` is a valid Counterfactual structure if it is either a single
  element (including tuples) or is an unnested dictionary (with string keys). If
  `element_type` is set, the function will also validate that all elements are
  of the correct type.

  Raises:
    TypeError: If `struct` is neither a single element (including a tuple) nor a
      dict.
    ValueError: If `struct` is a dict with non-string keys.
    ValueError: If `struct` is a dict with values that are not single elements
      (including tuples).
  """
  # pyformat: enable
  if _is_counterfactual_element(struct, element_type):
    return  # Valid single Counterfactual element.

  err_msg = f"`{struct_name}` is not a recognized Counterfactual structure."
  # If struct is not a counterfactual_element, then it should be a dict. If not,
  # raise an error.
  if not isinstance(struct, dict):
    accepted_types_msg = "a single unnested element (or tuple)"
    if element_type is not None:
      accepted_types_msg += " of type {}".format(element_type)
    accepted_types_msg += ", or a dict"
    raise TypeError(
        f"{err_msg} It should have a type of one of: {accepted_types_msg}. "
        f"Given: {type(struct)}")

  # Validate that keys are strings if struct is a dict.
  if not all([isinstance(key, str) for key in struct.keys()]):
    raise ValueError(
        f"{err_msg} If `{struct_name}` is a dict, it must contain only string "
        f"keys, given keys: {list(struct.keys())}")

  # Validate that values are all single Counterfactual elements.
  if not all([
      _is_counterfactual_element(element, element_type)
      for element in struct.values()
  ]):
    err_msg += "If it is a dict, it must be unnested"
    if element_type is not None:
      err_msg += f" and contain only elements of type {element_type}"
    raise ValueError(f"{err_msg}. Given: {struct}")
