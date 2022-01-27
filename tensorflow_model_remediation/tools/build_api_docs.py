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

"""Generates API docs for TensorFlow Model Remediation."""
import os

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_model_remediation as tfmr

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/tensorflow_model_remediation_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/model-remediation/tree/main/tensorflow_model_remediation/",
    "The URL prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path",
                    "responsible-ai/model_remediation/api_docs/python",
                    "Path prefix in the _toc.yaml")


def execute(output_dir, code_url_prefix, search_hints, site_path):
  """Builds API docs for TensorFlow Model Remediation."""

  # Hide `Model` methods with a few exceptions.
  for cls in [tf.Module, tf.keras.layers.Layer, tf.keras.Model]:
    doc_controls.decorate_all_class_attributes(
        decorator=doc_controls.do_not_doc_in_subclasses,
        cls=cls,
        skip=["__init__", "save", "compile", "call"])

  # Hide `Loss` methods with a few exceptions.
  for cls in [tf.keras.losses.Loss]:
    doc_controls.decorate_all_class_attributes(
        decorator=doc_controls.do_not_doc_in_subclasses,
        cls=cls,
        skip=["__init__", "call"])

  # Hide `MinDiffLoss` and `MinDiffKernel` __call__ method.
  for cls in [
      tfmr.min_diff.losses.MinDiffLoss, tfmr.min_diff.losses.MinDiffKernel
  ]:
    doc_controls.decorate_all_class_attributes(
        decorator=doc_controls.do_not_doc_in_subclasses,
        cls=cls,
        skip=["__init__"])

  # Get around the decorator on Layer.call
  delattr(tf.keras.layers.Layer.call,
          "_tf_docs_tools_for_subclass_implementers")

  # Delete common module when documenting. There is nothing there for users
  # quite yet.
  del tfmr.common

  try:
    del tfmr.tools
  except AttributeError:
    pass

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Model Remediation",
      py_modules=[("model_remediation", tfmr)],
      base_dir=os.path.dirname(tfmr.__file__),
      search_hints=search_hints,
      code_url_prefix=code_url_prefix,
      site_path=site_path,
      callbacks=[
          public_api.explicit_package_contents_filter,
          public_api.local_definitions_filter
      ])

  doc_generator.build(output_dir)


def main(unused_argv):
  execute(FLAGS.output_dir, FLAGS.code_url_prefix, FLAGS.search_hints,
          FLAGS.site_path)


if __name__ == "__main__":
  app.run(main)
