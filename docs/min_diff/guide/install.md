# Install Model Remediation

You have a few options to download and start using TensorFlow Model Remediation:

*    To start learning what the library contains and how it works, run the
     [tutorial](link to tutorial).
*    To download on a local machine, install the `tensorflow-model-remediation`
     pip package.
*    If your machine has a unique configuration, you can build your package from
     source.

## Installing the pip package

To install the pip package run the following command:
```shell
pip install --upgrade tensorflow-model-remediation
```

You will need TensorFlow 2.0 or higher.

## Installing from source

First, clone the github repo:

```shell
git clone https://github.com/tensorflow/model_remediation.git
```

Build the pip package from source (after any modifications if necessary):
```shell
python setup.py sdist bdist_wheel --universal --release
```

Finally, install your locally built package:
```shell
pip install --user --upgrade /path/to/pkg.whl
```
