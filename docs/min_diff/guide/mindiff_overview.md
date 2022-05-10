# Overview

MinDiff is a model remediation technique that seeks to equalize two distributions.
In practice, it can be used to balance error rates across different slices of
your data by penalizing distributional differences.

Typically, you apply MinDiff when trying to ensure group fairness, such as
minimizing the difference in either
false positive rate (FPR) or false negative rate (FNR) between a slice of data
belonging to a sensitive class and a better-performing slice. For in-depth
discussion of fairness metrics, review the literature on this subject.[^1][^2][^3]

## How does MinDiff work?

Given two sets of examples from our dataset, MinDiff penalizes the model during
training for differences in the distribution of scores between the two sets.
The less distinguishable the two sets are based on prediction scores, the
smaller the penalty that will be applied.

The penalty is applied by adding a component to the loss that the model is using
for training. It can be thought of as a measurement of the difference in
distribution of model predictions. As the model trains, it tries to minimize
the penalty by bringing the distributions closer together, as shown in the
graphs below.

![MinDiff comparison graph](/responsible_ai/model_remediation/min_diff/guide/images/mindiff_graphs.svg)


Applying MinDiff may come with tradeoffs with respect to performance on the original
task. MinDiff can be effective while not deteriorating
performance beyond product needs, but the decision to balance between performance
and effectiveness of MinDiff
should be made deliberately by the product owner. For examples showing how to implement
MinDiff, see [the model remediation case study notebook](/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras).


## Resources

* For a tutorial on applying MinDiff on a text classification model, see
  [MinDiff Keras notebook](/responsible_ai/model_remediation/min_diff/tutorials/min_diff_keras).

* For a blog post on MinDiff on the TensorFlow blog, see
  [Applying MinDiff to improve model blog post](https://blog.tensorflow.org/2020/11/applying-mindiff-to-improve-model.html).

* For the full Model Remediation library, see the
  [model-remediation Github repo](https://github.com/tensorflow/model-remediation).


[^1]: Dwork, C., Hardt, M., Pitassi, T., Reingold, O., Zemel, R. (2011).
      <a href="https://arxiv.org/abs/1104.3913">Fairness Through Awareness.</a>

[^2]: Hardt, M., Price, E., Srebro, N. (2016). <a href="https://arxiv.org/abs/1610.02413">
      Equality of Opportunity in Supervised Learning.</a>

[^3]: Chouldechova, A. (2016). <a href="https://arxiv.org/abs/1610.07524">
      Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.</a>
