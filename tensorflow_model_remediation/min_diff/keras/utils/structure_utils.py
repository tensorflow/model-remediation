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

"""Utils to validate and manipulate MinDiff structures (single elems or dicts).

This module is a customized version of `tf.nest` with the main difference being
that tuples are considered single elements rather than structs that can be
unpacked.
"""

import tensorflow as tf


def _flatten_min_diff_structure(struct, run_validation=False):
  # pyformat: disable
  """Flattens a MinDiff structure after optionally validating it.

  Arguments:
    struct: structure to be flattened. Must be a single element (including a
      tuple) or an unnested dict.
    run_validation: Boolean indicating whether to run validation. If `True`
      `validate_min_diff_structure` will be called on `struct`.

  Has similar behavior to `tf.nest.flatten` with the exception that tuples will
  be considered as single elements instead of structures to be flattened. See
  `tf.nest.flatten` documentation for additional details on behavior.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    ValueError: If struct is not a valid MinDiff structure (a single element
      including a tuple, or a dict).
  """
  # pyformat: enable

  if run_validation:
    validate_min_diff_structure(struct)

  if isinstance(struct, dict):
    return [struct[key] for key in sorted(struct.keys())]

  return [struct]  # Wrap in a list if not nested.


def _pack_min_diff_sequence_as(struct, flat_sequence):
  # pyformat: disable
  """Pack `flat_sequence` into the same structure as `struct`.

  Arguments:
    struct: structure that `flat_sequence` will be packed as. Must be a single
      element (including a tuple) or an unnested dict.
    flat_sequence: Flat sequence of elements to be packed.

  Has similar behavior to `tf.nest.pack_sequence_as` with the exception that
  tuples in `struct` will be considered as single elements. See
  `tf.nest.pack_sequence_as` documentation for additional details on behavior.

  Returns:
    `flat_sequence` converted to have the same structure as `struct`.

  Raises:
    ValueError: If `flat_sequence` has a different number of elements from
      `struct`. (Note: if `struct` is a dict, keys are used to count elements).
    ValueError: If `struct` is not a single element (including a tuple) or a
      dict. (Note: If `struct` is a nested dict, the nested values will be
      ignored).
  """
  # pyformat: enable
  if _is_min_diff_element(struct):
    if len(flat_sequence) != 1:
      raise ValueError(
          "The target structure is of type: {}\n\nHowever the input "
          "structure is a sequence ({}) of length {}: {}.".format(
              type(struct), type(flat_sequence), len(flat_sequence),
              flat_sequence))
    return flat_sequence[0]

  if isinstance(struct, dict):
    ordered_keys = sorted(struct.keys())
    if len(flat_sequence) != len(ordered_keys):
      raise ValueError(
          "Could not pack sequence. Dict had {} keys, but flat_sequence had {} "
          "element(s).  Structure: {}, flat_sequence: {}.".format(
              len(ordered_keys), len(flat_sequence), struct, flat_sequence))

    return {k: v for k, v in zip(ordered_keys, flat_sequence)}

  # If the code reaches here, then `struct` is not a valid MinDiff structure.
  # We call `validate_min_diff_structure` to raise the appropriate error.
  validate_min_diff_structure(struct)


def _assert_same_min_diff_structure(struct1, struct2):
  # pyformat: disable
  """Asserts that the two MinDiff structures are the same.

  Arguments:
    struct1: First MinDiff structure. Must be a single element (including a
      tuple) or an unnested dict.
    struct2: Second MinDiff structure. Must be a single element (including a
      tuple) or an unnested dict.

  Has similar behavior to `tf.nest.assert_same_structure` with the exception
  that tuples will be considered as single elements and not examined further.
  See `tf.nest.assert_same_structure` documentation for additional details on
  behavior.

  Raises:
    ValueError: If either `struct1` or `struct2` is invalid (see
      `validate_min_diff_structure` for details).
    ValueError: If `struct1` and `struct2` do not have the same structure.
  """
  # pyformat: enable

  # validate structures.
  validate_min_diff_structure(struct1)
  validate_min_diff_structure(struct2)

  def _err_msg(specifics_template, use_dict_keys=False):
    struct1_str = "type={} str={}".format(type(struct1), struct1)
    struct2_str = "type={} str={}".format(type(struct2), struct2)

    err_msg = ("The two structures don't have the same structure:\n\n"
               "First structure: {}\n\n"
               "Second structure: {}\n\n".format(struct1_str, struct2_str))
    err_msg += "More Specifically: "
    if use_dict_keys:
      err_msg += specifics_template.format(
          list(struct1.keys()), list(struct2.keys()))
    else:
      err_msg += specifics_template.format(struct1_str, struct2_str)
    return err_msg

  # Assert same structure if they are a dict.
  if isinstance(struct1, dict):
    if not isinstance(struct2, dict):
      raise ValueError(
          _err_msg("Substructure \"{}\" is a dict, while "
                   "substructure \"{}\" is not."))
    if set(struct1.keys()) != set(struct2.keys()):
      raise ValueError(
          _err_msg(
              "The two dictionaries don't have the same set of keys. First "
              "structure has keys: {}, while second structure has keys: {}.",
              use_dict_keys=True))
    return  # Dicts match.

  # If struct1 is not a nested structure, struct2 should not be either.
  if isinstance(struct2, dict):
    raise ValueError(
        _err_msg("Substructure \"{1}\" is a dict, while "
                 "substructure \"{0}\" is not."))


def _is_min_diff_element(element, element_type=None):
  """Returns `True` if `element` is a MinDiff element and `False` otherwise."""
  if element_type is not None:
    if not isinstance(element_type, type):
      raise TypeError(
          "`element_type` should be a class corresponding to expected type. "
          "Instead an object instance was given: {}".format(element_type))
    # Return False if element is of the wrong type (if indicated).
    if not isinstance(element, element_type):
      return False

  is_single_element = not tf.nest.is_nested(element)
  is_tuple = isinstance(element, tuple)
  return is_single_element or is_tuple


def validate_min_diff_structure(struct,
                                struct_name="struct",
                                element_type=None):
  # pyformat: disable
  """Validates that `struct` is a valid MinDiff structure.

  Arguments:
    struct: Structure that will be validated.
    struct_name: Name of struct, used only for error messages.
    element_type: Type of elements. If `None`, types are not checked.

  A `struct` is a valid MinDiff structure if it is either a single element
  (including tuples) or is an unnested dictionary (with string keys). If
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
  if _is_min_diff_element(struct, element_type):
    return  # Valid single MinDiff element.

  err_msg = "`{}` is not a recognized MinDiff structure.".format(struct_name)
  # If struct is not a min_diff_element, then it should be a dict. If not, raise
  # an error.
  if not isinstance(struct, dict):
    accepted_types_msg = "a single unnested element (or tuple)"
    if element_type is not None:
      accepted_types_msg += " of type {}".format(element_type)
    accepted_types_msg += ", or a dict"
    raise TypeError("{} It should have a type of one of: {}. Given: {}".format(
        err_msg, accepted_types_msg, type(struct)))

  # Validate that keys are strings if struct is a dict.
  if not all([isinstance(key, str) for key in struct.keys()]):
    raise ValueError(
        "{} If `{}` is a dict, it must contain only string keys, given keys: {}"
        .format(err_msg, struct_name, list(struct.keys())))

  # Validate that values are all single MinDiff elements.
  if not all([
      _is_min_diff_element(element, element_type)
      for element in struct.values()
  ]):
    err_msg += "If it is a dict, it must be unnested"
    if element_type is not None:
      err_msg += " and contain only elements of type {}".format(element_type)
    raise ValueError("{}. Given: {}".format(err_msg, struct))
