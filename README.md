# TensorFlow Model Remediation

TensorFlow Model Remediation is a library that provides solutions for machine
learning practitioners working to create and train models in a way that reduces
or eliminates user harm resulting from underlying performance biases.

[![PyPI version](https://badge.fury.io/py/tensorflow-model-remediation.svg)](https://badge.fury.io/py/tensorflow-model-remediation)
[![Tutorial](https://img.shields.io/badge/doc-tutorial-blue.svg)](https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras)
[![Overview](https://img.shields.io/badge/doc-overview-blue.svg)](https://www.tensorflow.org/responsible_ai/model_remediation)

## Installation

You can install the package from `pip`:

```shell
$ pip install tensorflow-model-remediation
```

Note: Make sure you are using TensorFlow 2.x.

## Documentation

This library will ultimately contain a collection of techniques for addressing
a wide range of concerns. For now it contains a single technique, MinDiff,
which can help reduce performance gaps between example subgroups.


We recommend starting with the
[overview guide](https://www.tensorflow.org/responsible_ai/model_remediation)
or trying it interactively in our
[tutorial notebook](https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras).



```python
from tensorflow_model_remediation import min_diff
import tensorflow as tf

# Start by defining a Keras model.
original_model = ...

# Set the MinDiff weight and choose a loss.
min_diff_loss = min_diff.losses.MMDLoss()
min_diff_weight = 1.0  # Hyperparamater to be tuned.

# Create a MinDiff model.
min_diff_model = min_diff.keras.MinDiffModel(
    original_model, min_diff_loss, min_diff_weight)

# Compile the MinDiff model as you normally would do with the original model.
min_diff_model.compile(...)

# Create a MinDiff Dataset and train the min_diff_model on it.
min_diff_model.fit(min_diff_dataset, ...)
```

#### *Disclaimers*

*If you're interested in learning more about responsible AI practices, including*
*fairness, please see Google AI's [Responsible AI Practices](https://ai.google/education/responsible-ai-practices).*

*`tensorflow/model_remediation` is Apache 2.0 licensed. See the
[`LICENSE`](LICENSE) file.*
