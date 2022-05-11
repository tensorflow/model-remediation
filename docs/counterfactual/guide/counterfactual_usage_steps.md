# Using Counterfactual Logit Pairing

Once you have determined that Counterfactual Logit Pairing (CLP) is the
appropriate technique for your use case, you can apply it by taking the
following steps:

1. [Create an instance of `CounterfactualPackedInputs`](#counterfactual_dataset)
    with the original and counterfactual data.
2. [Measure](#measure_flip_count_and_flip_rate) the flip rate and flip
    count to determine if intervention is needed.
3. If intervention is needed, [pass](#apply_clp) in the original input data,
   counterfactual data, original model, and couterfactual loss to the
   counterfactual model.
4. Assess the impact of CLP by measuring the flip rate and flip count.

To see an example of applying the CLP to a Keras model, see the
[Use Counterfactual Logit Pairing with Keras tutorial](/responsible_ai/model_remediation/counterfactual/guide/counterfactual_keras).

## Create an instance of `CounterfactualPackedInputs` {: #counterfactual_dataset}

To create the counterfactual dataset, start by determining the terms and
features you want to assess that, when removed or replaced, may alter the
prediction of your model.

Once you understand the terms and features to assess on, you will need to
create an instance of `CounterfactualPackedInputs`, which includes the original
input and counterfactual data. The original input should be the dataset
you used to train your Keras model. Counterfactual data has an `original_x`
value, a `counterfactual_x` value and a `counterfactual_sample_weight`. The
counterfactual value should be nearly identical to the original value with
the difference being one or more of the sensitive attributes are removed or
replaced. The quality of the counterfactual dataset is important as it is used
to pair the loss function between the original value and the counterfactual
value with the goal of assuring that the model’s prediction doesn’t change
when the sensitive attribute is different.

For details on how to develop this counterfactual dataset, see the notebook
on [creating a custom counterfactual dataset](https://tensorflow.org/responsible_ai/model_remediation/counterfactual/guide/creating_a_custom_counterfactual_dataset).

## Measure flip count and flip rate {: #measure_flip_count_and_flip_rate}

A *flip* is defined as a classifier giving a different decision when the
sensitive attribute referenced in the example changes. It captures the
situation where a classifier changes its prediction in the presence, absence,
or change of an identity attribute. A more continuous metric should be used
when assessing the real value (score) of a classifier.

#### Flip Count

Flip count measures the number of times the classifier gives a different
decision if the identity term in a given example were changed.
* *Overall Flip Count*: Total flips of a prediction from positive to negative
  and vice versa.
* *Positive to Negative Prediction Flip Count*: Number of flips where the
  prediction label changed from positive to negative.
* *Negative to Positive Prediction Flip Count*: Number of flips where the
  prediction label changed from negative to positive.

#### Flip Rate

Flip rate measures the probability that the classifier gives a different
decision if the identity term in a given example were changed.

* *Overall Flip Rate*: Total flip count over the total number of examples
* *Positive to Negative Prediction Flip Rate*: Positive to negative flip count
  over positive examples in counterfactual dataset
* *Negative to Positive Prediction Flip Rate*: Negative to positive flip count
  over negative examples in counterfactual dataset

After calculating the flip rate and flip count with [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide),
you can determine if the classifier is making a different prediction based on a
sensitive attribute within the data. You can use the example count and
confidence intervals to determine if you have sufficient data to apply CLP
and draw conclusions from the flip rate. A high flip rate and flip count are
indicative of this behavior occurring and can be used to decide whether CLP
is appropriate for your use case. This decision is specific to your model and
depends on factors such as the harm that may be caused to end users and the
product that the model is used in.

## Apply Counterfactual Logit Pairing to your Keras Model {: #apply_clp}

To use CLP, you need the original Keras model you're
looking to remediate, the original training dataset, and the counterfactual
dataset. Determine what [`counterfactual loss`](https://www.tensorflow.org/responsible_ai/model_remediation/api_docs/python/model_remediation/counterfactual/losses/CounterfactualLoss)
should be applied for the logit pairing. With this, you can build the
Counterfactual model with the desired counterfactual loss function and loss
function from your original model.

After applying CLP, you should calculate the flip rate and flip count, and any
changes in other metrics such as overall accuracy to measure the improvement
that resulted from applying this technique.
