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

"""Documentation utils for the tensorflow.org website.

Note: To be detected, doc decorators should be applied between descriptors
and other decorators.

```py
class A:

  @staticmethod
  @tensorflow_model_remediation.common.docs.doc_private
  @other_decorator
  def f():
    pass
```

"""

from tensorflow.tools.docs import doc_controls


def _no_op_decorator(obj):
  return obj


def _get_safe_decorator(decorator_name):

  try:
    return getattr(doc_controls, decorator_name)
  except AttributeError:

    return _no_op_decorator


do_not_generate_docs = _get_safe_decorator("do_not_generate_docs")
doc_private = _get_safe_decorator("doc_private")
do_not_doc_in_subclasses = _get_safe_decorator("do_not_doc_in_subclasses")
doc_in_current_and_subclasses = _get_safe_decorator(
    "doc_in_current_and_subclasses")
