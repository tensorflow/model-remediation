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

"""Tests for structure_utils functions."""

import tensorflow as tf

from tensorflow_model_remediation.min_diff.keras.utils import structure_utils


class ValidateStructureTest(tf.test.TestCase):

  def testValidateSingleElement(self):
    elem = "element"  # String
    structure_utils.validate_min_diff_structure(elem)

    elem = 3  # int
    structure_utils.validate_min_diff_structure(elem)

    elem = ("a", "b", "c")
    structure_utils.validate_min_diff_structure(elem)

  def testValidateSingleElementWithTypeCheck(self):
    elem = "element"
    structure_utils.validate_min_diff_structure(elem, element_type=str)

    with self.assertRaisesRegex(
        TypeError, "not a recognized MinDiff structure.*should have a type of"
        ".*single unnested element.*of type.*int.*str"):
      structure_utils.validate_min_diff_structure(elem, element_type=int)

    elem = 3
    structure_utils.validate_min_diff_structure(elem, element_type=int)

    with self.assertRaisesRegex(
        TypeError, "not a recognized MinDiff structure.*should have a type of"
        ".*single unnested element.*of type.*str.*.*int"):
      structure_utils.validate_min_diff_structure(elem, element_type=str)

  def testValidateRaisesErrorWithBadElementType(self):
    with self.assertRaisesRegex(
        TypeError, "element_type.*expected type.*object instance was given.*a"):
      structure_utils.validate_min_diff_structure("elem", element_type="a")

  def testValidateDict(self):
    elem = "element"
    struct = {"a": elem, "b": elem, "c": elem}
    structure_utils.validate_min_diff_structure(struct)

    # Assert raises an error if dict is not simple.
    struct = {"a": elem, "b": {"d": elem}, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested.*{.*{.*}.*}"):
      structure_utils.validate_min_diff_structure(struct, struct_name="name")

    # Assert raises an error if dict has non string keys.
    struct = {"a": elem, 3: elem, "c": elem}
    with self.assertRaisesRegex(
        ValueError, r"name.*must contain only string keys.*\['a', 3, 'c'\]"):
      structure_utils.validate_min_diff_structure(struct, struct_name="name")

  def testValidateDictWithTypeCheck(self):
    elem = "element"
    struct = {"a": elem, "b": elem, "c": elem}
    structure_utils.validate_min_diff_structure(struct, element_type=str)

    # Assert raises an error if one or more elements are the wrong type.
    # A single element is the wrong type.
    struct = {"a": elem, "b": 2, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested and contain only "
        "elements of type.*str.*{.*2.*}"):
      structure_utils.validate_min_diff_structure(
          struct, struct_name="name", element_type=str)

    # All elements are the (same) wrong type.
    struct = {"a": elem, "b": 2, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested and contain only "
        "elements of type.*str.*{.*2.*}"):
      structure_utils.validate_min_diff_structure(
          struct, struct_name="name", element_type=str)


class FlattenStructureTest(tf.test.TestCase):

  def testFlattenSingleElement(self):
    elem = "element"
    flat_elem = structure_utils._flatten_min_diff_structure(elem)
    # Single element should be put into a list of size 1.
    self.assertAllEqual(flat_elem, [elem])

  def testFlattenDict(self):
    struct = {"a": "a", "b": "b", "c": "c"}
    flat_elem = structure_utils._flatten_min_diff_structure(struct)
    # Dict should be flattened into a list in order key order.
    self.assertAllEqual(flat_elem, ["a", "b", "c"])

  def testFlattenDictWithTuples(self):
    struct = {"a": ("a",), "b": ("b", "b"), "c": ("c",)}
    flat_elem = structure_utils._flatten_min_diff_structure(struct)
    # Dict should be flattened into a list in order key order.
    self.assertAllEqual(flat_elem, [("a",), ("b", "b"), ("c",)])

  def testValidateOptionallyCalled(self):
    struct = {"a": "a", "b": {"d": "b"}, "c": "c"}  # Bad struct (nested)

    # By default validation is not run.
    structure_utils._flatten_min_diff_structure(struct)

    with self.assertRaisesRegex(ValueError, "dict.*must be unnested"):
      structure_utils._flatten_min_diff_structure(struct, run_validation=True)


class AssertSameStructureTest(tf.test.TestCase):


  def setUp(self):
    super().setUp()
    self._err_msg_intro_regex_template = (
        "The two structures don't have the same structure.*\n\n.*First "
        "structure.*{}.*\n\n.*Second structure.*{}.*\n\nMore Specifically.*")

  def testValidatesStructures(self):
    struct = {"a": "a", "b": "b", "c": "c"}
    bad_struct = {"a": "a", "b": {"d": "b"}, "c": "c"}  # Bad struct (nested)

    # First struct is invalid.
    with self.assertRaisesRegex(ValueError, "dict.*must be unnested"):
      structure_utils._assert_same_min_diff_structure(bad_struct, struct)

    # Second struct is invalid.
    with self.assertRaisesRegex(ValueError, "dict.*must be unnested"):
      structure_utils._assert_same_min_diff_structure(struct, bad_struct)

  def testDictsHaveSameStructure(self):
    struct1 = {"a": "a", "b": "b", "c": "c"}
    struct2 = {"a": "d", "b": "e", "c": "f"}

    structure_utils._assert_same_min_diff_structure(struct1, struct2)

    struct1_str = "{'a': 'a', 'b': 'b', 'c': 'c'}"
    bad_dict = "not a dict"
    with self.assertRaisesRegex(
        ValueError,
        self._err_msg_intro_regex_template.format(struct1_str, bad_dict) +
        "{}.*is a dict.*{}.*is not".format(struct1_str, bad_dict)):
      structure_utils._assert_same_min_diff_structure(struct1, bad_dict)
    # Assert also raises in reverse order.
    with self.assertRaisesRegex(
        ValueError,
        self._err_msg_intro_regex_template.format(bad_dict, struct1_str) +
        "{}.*is a dict.*{}.*is not".format(struct1_str, bad_dict)):
      structure_utils._assert_same_min_diff_structure(bad_dict, struct1)

    struct1_keys_str = r"\['a', 'b', 'c'\]"
    struct2_subset = {"a": "d", "b": "e"}
    struct2_subset_str = "{'a': 'd', 'b': 'e'}"
    struct2_subset_keys_str = r"\['a', 'b'\]"
    with self.assertRaisesRegex(
        ValueError,
        self._err_msg_intro_regex_template.format(struct1_str,
                                                  struct2_subset_str) +
        "don't have the same set of keys.*{}.*{}".format(
            struct1_keys_str, struct2_subset_keys_str)):
      structure_utils._assert_same_min_diff_structure(struct1, struct2_subset)
    # Assert also raises in reverse order.
    with self.assertRaisesRegex(
        ValueError,
        self._err_msg_intro_regex_template.format(struct2_subset_str,
                                                  struct1_str) +
        "don't have the same set of keys.*{}.*{}".format(
            struct2_subset_keys_str, struct1_keys_str)):
      structure_utils._assert_same_min_diff_structure(struct2_subset, struct1)

    struct2_alt = {"a": "d", "b": "e", "f": "f"}
    struct2_alt_str = "{'a': 'd', 'b': 'e', 'f': 'f'}"
    struct2_alt_keys_str = r"\['a', 'b', 'f'\]"
    with self.assertRaisesRegex(
        ValueError,
        self._err_msg_intro_regex_template.format(struct1_str, struct2_alt_str)
        + "don't have the same set of keys.*{}.*{}".format(
            struct1_keys_str, struct2_alt_keys_str)):
      structure_utils._assert_same_min_diff_structure(struct1, struct2_alt)

  def testDictsWithTuplesHaveSameStructure(self):
    struct1 = {"a": "a", "b": "b", "c": "c"}
    struct2 = {"a": ("d",), "b": ("e", "e"), "c": ("f",)}

    structure_utils._assert_same_min_diff_structure(struct1, struct2)


class PackSequenceAsStructureTest(tf.test.TestCase):

  def testPackAsSingleElement(self):
    elem = "elem"
    struct = "single element"
    seq = [elem]
    res = structure_utils._pack_min_diff_sequence_as(struct, seq)
    self.assertEqual(res, elem)

    short_seq = []
    with self.assertRaisesRegex(
        ValueError, "structure is of type.*str(.|\n)*"
        r"structure is a sequence.*list.*0.*\[\]"):
      _ = structure_utils._pack_min_diff_sequence_as(struct, short_seq)

    long_seq = [1, 2]
    with self.assertRaisesRegex(
        ValueError, "structure is of type.*str(.|\n)*"
        r"structure is a sequence.*list.*2.*\[1, 2\]"):
      _ = structure_utils._pack_min_diff_sequence_as(struct, long_seq)

  def testPackAsDict(self):
    struct = {"k1": "elem", "k2": "elem"}
    seq = [1, 2]
    res = structure_utils._pack_min_diff_sequence_as(struct, seq)
    self.assertDictEqual(res, {"k1": 1, "k2": 2})
    # Passing in the reversed sequence should result in the key values be
    # swapped.
    res = structure_utils._pack_min_diff_sequence_as(struct,
                                                     list(reversed(seq)))
    self.assertDictEqual(res, {"k1": 2, "k2": 1})

    short_seq = [1]
    with self.assertRaisesRegex(
        ValueError, "Dict had 2 keys.*flat_sequence "
        r"had 1 element\(s\).*(k[12].*){2}.*\[1\]"):
      _ = structure_utils._pack_min_diff_sequence_as(struct, short_seq)

    long_seq = [1, 2, 3]
    with self.assertRaisesRegex(
        ValueError, "Dict had 2 keys.*flat_sequence had"
        r" 3 element\(s\).*(k[12].*){2}.*\[1, 2, 3\]"):
      _ = structure_utils._pack_min_diff_sequence_as(struct, long_seq)

  def testInvalidStructRaisesError(self):
    struct = [1, 2, 3]  # Not a valid MinDiff structure.
    with self.assertRaisesRegex(
        TypeError, "not a recognized MinDiff structure.*should have a type of"
        ".*single unnested element.*list"):
      _ = structure_utils._pack_min_diff_sequence_as(struct, [])


if __name__ == "__main__":
  tf.test.main()
