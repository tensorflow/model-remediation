# Install Model Remediation

You have a few options to download and start using TensorFlow Model Remediation:

*    To start learning what the library contains and how it works, run a
     tutorial for the technique that you're interested in. See the
     [Intervening on the model](/responsible_ai/model_remediation#intervening-on-the-model)
     section for the list of techniques and links to tutorials.
*    To download on a local machine, install the
     [`tensorflow-model-remediation`](https://pypi.org/project/tensorflow-model-remediation/){: .external}
     pip package.
*    If your machine has a unique configuration, you can build your package from
     [source](https://github.com/tensorflow/model-remediation).

## Installing the pip package

To install the pip package run the following command:
```shell
$ pip install --upgrade tensorflow-model-remediation
```

This package requires TensorFlow 2.0 or higher.

## Installing from source

First, clone the github repo:

```shell
$ git clone https://github.com/tensorflow/model-remediation.git
```

Build the pip package from source (after any modifications if necessary):
```shell
$ python3 setup.py sdist bdist_wheel
```

Finally, install your locally built package:
```shell
$ pip install --upgrade dist/tensorflow_model_remediation-*.whl
```
