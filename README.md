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

This library contains a collection of machine learning remediation techniques
for addressing potential bias in a model.

Currently TensorFlow Model Remediation contains the below techniques:

*   MinDiff technique: Typically used to ensure that a model predicts the
    preferred label equally well for all values of a sensitive attribute.
    Helpful when trying to achieve (equality of
    opportunity)[https://developers.google.com/machine-learning/glossary/fairness#equality-of-opportunity].

*   Counterfactual Logit Pairing technique: Typically used to ensure that a
    model’s prediction does not change between “counterfactual pairs”, where the
    sensitive attribute referenced in a feature is different. Helpful when
    trying to achieve
    [counterfactual fairness](https://developers.google.com/machine-learning/glossary/fairness#counterfactual-fairness).

We recommend starting with the
[overview guide](https://www.tensorflow.org/responsible_ai/model_remediation) to
get an idea of TensorFlow Model Remediation. Next try one of our interactive
guides like the

[MinDiff tutorial notebook](https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras).

[Counterfactual tutorial notebook](https://www.tensorflow.org/responsible_ai/model_remediation/counterfactual/guide/counterfactual_keras).


```python

import tensorflow_model_remediation as tfmr

import tensorflow as tf

# Start by defining a Keras model.

original_model = ...

# Next pick the remediation technique you'd like to use. For example, a
# MinDiff implementation might look like the below:
# Set the MinDiff weight and choose a loss.

min_diff_loss = tfmr.min_diff.losses.MMDLoss()

min_diff_weight = 1.0  # Hyperparamater to be tuned.

# Create a MinDiff model.

min_diff_model = tfmr.min_diff.keras.MinDiffModel(

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
