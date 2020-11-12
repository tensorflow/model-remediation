# TensorFlow Model Remediation

Model Remediation is a library that provides solutions for machine learning
practitioners working to create and train models in a way that reduces or
eliminates user harm resulting from underlying performance biases.

[![PyPI version](https://badge.fury.io/py/tensorflow-model-remediation.svg)](https://badge.fury.io/py/tensorflow-model-remediation)

## Documentation

To install and use Model Remediation, start with our
[**getting started guide**](https://www.tensorflow.org/responsible_ai/model_remediation) and try
it interactively in a
[Colab notebook](https://github.com/tensorflow/model-remediation/blob/master/docs/examples/min_diff_keras.ipynb).

To get started with Model Remediation:

```python
# !pip install tensorflow-model-remediation
from fairness_indicators.examples import util
from tensorflow_model_remediation import min_diff
import tensorflow as tf

# Start with a given Keras model.
original_model = util.create_keras_sequential_model()

# Set the strenth of MinDiff and a given loss.
min_diff_weight = 1.0
min_diff_loss = min_diff.losses.MMDLoss()

# Create a MinDiff model.
min_diff_model = min_diff.keras.MinDiffModel(
    original_model, min_diff_loss, min_diff_strength)

# Compile the MinDiff model as you normally would do with Keras.
min_diff_model.compile(...)
```

#### *Disclaimers*

*If you're interested in learning more about responsible AI practices, including*
*fairness, please see Google AI's [Responsible AI Practices](https://ai.google/education/responsible-ai-practices).*

*`tensorflow/model_remediation` is Apache 2.0 licensed. See the
[`LICENSE`](LICENSE) file.*
