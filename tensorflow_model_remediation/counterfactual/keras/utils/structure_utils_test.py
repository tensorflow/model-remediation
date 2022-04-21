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

from tensorflow_model_remediation.counterfactual.keras.utils import structure_utils


class ValidateStructureTest(tf.test.TestCase):

  def testValidateSingleElement(self):
    elem = "element"  # String
    structure_utils.validate_counterfactual_structure(elem)

    elem = 3  # int
    structure_utils.validate_counterfactual_structure(elem)

    elem = ("a", "b", "c")
    structure_utils.validate_counterfactual_structure(elem)

  def testValidateSingleElementWithTypeCheck(self):
    elem = "element"
    structure_utils.validate_counterfactual_structure(elem, element_type=str)

    with self.assertRaisesRegex(
        TypeError,
        "not a recognized Counterfactual structure.*should have a type of"
        ".*single unnested element.*of type.*int.*str"):
      structure_utils.validate_counterfactual_structure(elem, element_type=int)

    elem = 3
    structure_utils.validate_counterfactual_structure(elem, element_type=int)

    with self.assertRaisesRegex(
        TypeError,
        "not a recognized Counterfactual structure.*should have a type of"
        ".*single unnested element.*of type.*str.*.*int"):
      structure_utils.validate_counterfactual_structure(elem, element_type=str)

  def testValidateRaisesErrorWithBadElementType(self):
    with self.assertRaisesRegex(
        TypeError, "element_type.*expected type.*object instance was given.*a"):
      structure_utils.validate_counterfactual_structure(
          "elem", element_type="a")

  def testValidateDict(self):
    elem = "element"
    struct = {"a": elem, "b": elem, "c": elem}
    structure_utils.validate_counterfactual_structure(struct)

    # Assert raises an error if dict is not simple.
    struct = {"a": elem, "b": {"d": elem}, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested.*{.*{.*}.*}"):
      structure_utils.validate_counterfactual_structure(
          struct, struct_name="name")

    # Assert raises an error if dict has non string keys.
    struct = {"a": elem, 3: elem, "c": elem}
    with self.assertRaisesRegex(
        ValueError, r"name.*must contain only string keys.*\['a', 3, 'c'\]"):
      structure_utils.validate_counterfactual_structure(
          struct, struct_name="name")

  def testValidateDictWithTypeCheck(self):
    elem = "element"
    struct = {"a": elem, "b": elem, "c": elem}
    structure_utils.validate_counterfactual_structure(struct, element_type=str)

    # Assert raises an error if one or more elements are the wrong type.
    # A single element is the wrong type.
    struct = {"a": elem, "b": 2, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested and contain only "
        "elements of type.*str.*{.*2.*}"):
      structure_utils.validate_counterfactual_structure(
          struct, struct_name="name", element_type=str)

    # All elements are the (same) wrong type.
    struct = {"a": elem, "b": 2, "c": elem}
    with self.assertRaisesRegex(
        ValueError, "name.* is a dict.*must be unnested and contain only "
        "elements of type.*str.*{.*2.*}"):
      structure_utils.validate_counterfactual_structure(
          struct, struct_name="name", element_type=str)


class FlattenStructureTest(tf.test.TestCase):

  def testFlattenSingleElement(self):
    elem = "element"
    flat_elem = structure_utils._flatten_counterfactual_structure(elem)
    # Single element should be put into a list of size 1.
    self.assertAllEqual(flat_elem, [elem])

  def testFlattenDict(self):
    struct = {"a": "a", "b": "b", "c": "c"}
    flat_elem = structure_utils._flatten_counterfactual_structure(struct)
    # Dict should be flattened into a list in order key order.
    self.assertAllEqual(flat_elem, ["a", "b", "c"])

  def testFlattenDictWithTuples(self):
    struct = {"a": ("a",), "b": ("b", "b"), "c": ("c",)}
    flat_elem = structure_utils._flatten_counterfactual_structure(struct)
    # Dict should be flattened into a list in order key order.
    self.assertAllEqual(flat_elem, [("a",), ("b", "b"), ("c",)])

  def testValidateOptionallyCalled(self):
    struct = {"a": "a", "b": {"d": "b"}, "c": "c"}  # Bad struct (nested)

    # By default validation is not run.
    structure_utils._flatten_counterfactual_structure(struct)

    with self.assertRaisesRegex(ValueError, "dict.*must be unnested"):
      structure_utils._flatten_counterfactual_structure(
          struct, run_validation=True)


if __name__ == "__main__":
  tf.test.main()
